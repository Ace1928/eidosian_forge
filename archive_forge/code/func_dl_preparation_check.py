import contextlib
import io
import math
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.data_loader import SeedableRandomSampler, prepare_data_loader
from accelerate.state import AcceleratorState
from accelerate.test_utils import RegressionDataset, are_the_same_tensors
from accelerate.utils import (
def dl_preparation_check():
    state = AcceleratorState()
    length = 32 * state.num_processes
    dl = DataLoader(range(length), batch_size=8)
    dl = prepare_data_loader(dl, state.device, state.num_processes, state.process_index, put_on_device=True)
    result = []
    for batch in dl:
        result.append(gather(batch))
    result = torch.cat(result)
    print(state.process_index, result, type(dl))
    assert torch.equal(result.cpu(), torch.arange(0, length).long()), 'Wrong non-shuffled dataloader result.'
    dl = DataLoader(range(length), batch_size=8)
    dl = prepare_data_loader(dl, state.device, state.num_processes, state.process_index, put_on_device=True, split_batches=True)
    result = []
    for batch in dl:
        result.append(gather(batch))
    result = torch.cat(result)
    assert torch.equal(result.cpu(), torch.arange(0, length).long()), 'Wrong non-shuffled dataloader result.'
    if state.process_index == 0:
        print('Non-shuffled dataloader passing.')
    dl = DataLoader(range(length), batch_size=8, shuffle=True)
    dl = prepare_data_loader(dl, state.device, state.num_processes, state.process_index, put_on_device=True)
    result = []
    for batch in dl:
        result.append(gather(batch))
    result = torch.cat(result).tolist()
    result.sort()
    assert result == list(range(length)), 'Wrong shuffled dataloader result.'
    dl = DataLoader(range(length), batch_size=8, shuffle=True)
    dl = prepare_data_loader(dl, state.device, state.num_processes, state.process_index, put_on_device=True, split_batches=True)
    result = []
    for batch in dl:
        result.append(gather(batch))
    result = torch.cat(result).tolist()
    result.sort()
    assert result == list(range(length)), 'Wrong shuffled dataloader result.'
    if state.local_process_index == 0:
        print('Shuffled dataloader passing.')