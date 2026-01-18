import logging
import math
import os
from copy import deepcopy
import datasets
import evaluate
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from accelerate import Accelerator, DataLoaderConfiguration, DistributedType
from accelerate.data_loader import DataLoaderDispatcher
from accelerate.test_utils import RegressionDataset, RegressionModel, torch_device
from accelerate.utils import is_torch_xla_available, set_seed
def test_gather_for_metrics_drop_last():
    accelerator = Accelerator()
    per_device_batch_size = 5
    num_items = 10 * accelerator.num_processes + 1
    dataloader = DataLoader(range(num_items), batch_size=per_device_batch_size, drop_last=True)
    dataloader = accelerator.prepare(dataloader)
    iterator = iter(dataloader)
    next(iterator)
    batch = next(iterator)
    gathered_items = accelerator.gather_for_metrics(batch)
    num_expected_items = per_device_batch_size * accelerator.num_processes
    assert gathered_items.size(0) == num_expected_items, f'Expected number of items: {num_expected_items}, Actual: {gathered_items.size(0)}'