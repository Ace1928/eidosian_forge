import warnings
from typing import List
from unittest.mock import Mock
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from accelerate.accelerator import Accelerator, DataLoaderConfiguration
from accelerate.utils.dataclasses import DistributedType
def test_can_join_uneven_inputs():
    accelerator = create_accelerator(even_batches=False)
    model = torch.nn.Linear(1, 1)
    ddp_model = accelerator.prepare(model)
    dl = create_dataloader(accelerator, dataset_size=3, batch_size=1)
    batch_idxs = []
    with accelerator.join_uneven_inputs([ddp_model]):
        for batch_idx, batch in enumerate(dl):
            output = ddp_model(batch[0].float())
            loss = output.sum()
            loss.backward()
            batch_idxs.append(batch_idx)
    accelerator.wait_for_everyone()
    if accelerator.process_index == 0:
        assert batch_idxs == [0, 1]
    elif accelerator.process_index == 1:
        assert batch_idxs == [0]