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
def check_seedable_sampler():
    set_seed(42)
    train_set = RegressionDataset(length=10, seed=42)
    train_dl = DataLoader(train_set, batch_size=2, shuffle=True)
    config = DataLoaderConfiguration(use_seedable_sampler=True)
    accelerator = Accelerator(dataloader_config=config)
    train_dl = accelerator.prepare(train_dl)
    original_items = []
    for _ in range(3):
        for batch in train_dl:
            original_items.append(batch['x'])
    original_items = torch.cat(original_items)
    set_seed(42)
    train_dl.set_epoch(0)
    new_items = []
    for _ in range(3):
        for batch in train_dl:
            new_items.append(batch['x'])
    new_items = torch.cat(new_items)
    assert torch.allclose(original_items, new_items), 'Did not obtain the same items with the same seed and epoch.'