import warnings
from typing import List
from unittest.mock import Mock
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from accelerate.accelerator import Accelerator, DataLoaderConfiguration
from accelerate.utils.dataclasses import DistributedType
def test_can_disable_even_batches():
    accelerator = create_accelerator(even_batches=False)
    verify_dataloader_batch_sizes(accelerator, dataset_size=3, batch_size=1, process_0_expected_batch_sizes=[1, 1], process_1_expected_batch_sizes=[1])
    verify_dataloader_batch_sizes(accelerator, dataset_size=7, batch_size=2, process_0_expected_batch_sizes=[2, 2], process_1_expected_batch_sizes=[2, 1])