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
def check_seedable_sampler_in_batch_sampler_shard():
    set_seed(42)
    config = DataLoaderConfiguration(use_seedable_sampler=True)
    accelerator = Accelerator(dataloader_config=config)
    assert accelerator.num_processes > 1, 'This test requires more than one process.'
    dataloader = DataLoader(list(range(10)), batch_size=1, shuffle=True)
    prepared_data_loader = prepare_data_loader(dataloader=dataloader, use_seedable_sampler=True)
    target_sampler = prepared_data_loader.batch_sampler.batch_sampler.sampler
    assert isinstance(target_sampler, SeedableRandomSampler), 'Sampler in BatchSamplerShard is not SeedableRandomSampler.'