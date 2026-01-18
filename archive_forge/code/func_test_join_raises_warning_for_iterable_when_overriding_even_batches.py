import warnings
from typing import List
from unittest.mock import Mock
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from accelerate.accelerator import Accelerator, DataLoaderConfiguration
from accelerate.utils.dataclasses import DistributedType
def test_join_raises_warning_for_iterable_when_overriding_even_batches():
    accelerator = create_accelerator()
    model = torch.nn.Linear(1, 1)
    ddp_model = accelerator.prepare(model)
    create_dataloader(accelerator, dataset_size=3, batch_size=1, iterable=True)
    with warnings.catch_warnings(record=True) as w:
        with accelerator.join_uneven_inputs([ddp_model], even_batches=False):
            pass
        assert issubclass(w[-1].category, UserWarning)
        assert 'only supported for map-style datasets' in str(w[-1].message)