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
def custom_sampler_check():
    state = AcceleratorState()

    class CustomDataset(Dataset):

        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]

    class CustomBatchSampler:

        def __init__(self, dataset_length: int, batch_size: int, shuffle: bool=True):
            self.batch_size = batch_size
            self.data_index = np.arange(dataset_length)
            self.shuffle = shuffle

        def __iter__(self):
            num_batches = len(self)
            if self.shuffle:
                index = np.random.permutation(self.data_index)
            else:
                index = self.data_index
            output = np.array_split(index, num_batches)
            yield from output

        def __len__(self):
            return math.ceil(len(self.data_index) / self.batch_size)
    dataset = CustomDataset(range(32 * state.num_processes))
    sampler = CustomBatchSampler(len(dataset), batch_size=8)
    dl = DataLoader(dataset, batch_sampler=sampler)
    dl = prepare_data_loader(dl, state.device, state.num_processes, state.process_index)
    if hasattr(dl.batch_sampler, 'batch_sampler'):
        assert isinstance(dl.batch_sampler.batch_sampler, CustomBatchSampler), 'Custom sampler was changed after calling `prepare_data_loader`'
    else:
        assert isinstance(dl.batch_sampler, CustomBatchSampler), 'Custom sampler was changed after calling `prepare_data_loader`'