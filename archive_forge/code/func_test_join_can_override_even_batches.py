import warnings
from typing import List
from unittest.mock import Mock
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from accelerate.accelerator import Accelerator, DataLoaderConfiguration
from accelerate.utils.dataclasses import DistributedType
def test_join_can_override_even_batches():
    default_even_batches = True
    overridden_even_batches = False
    accelerator = create_accelerator(even_batches=default_even_batches)
    model = torch.nn.Linear(1, 1)
    ddp_model = accelerator.prepare(model)
    train_dl = create_dataloader(accelerator, dataset_size=3, batch_size=1)
    valid_dl = create_dataloader(accelerator, dataset_size=3, batch_size=1)
    with accelerator.join_uneven_inputs([ddp_model], even_batches=overridden_even_batches):
        train_dl_overridden_value = train_dl.batch_sampler.even_batches
        valid_dl_overridden_value = valid_dl.batch_sampler.even_batches
    assert train_dl_overridden_value == overridden_even_batches
    assert valid_dl_overridden_value == overridden_even_batches
    assert train_dl.batch_sampler.even_batches == default_even_batches
    assert valid_dl.batch_sampler.even_batches == default_even_batches