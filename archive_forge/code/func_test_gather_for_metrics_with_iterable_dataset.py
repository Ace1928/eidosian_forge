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
def test_gather_for_metrics_with_iterable_dataset():

    class DummyIterableDataset(IterableDataset):

        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            yield from self.data
    iterable_dataset = DummyIterableDataset(torch.as_tensor(range(30)))
    dataloader = DataLoader(iterable_dataset, batch_size=4)
    accelerator = Accelerator()
    prepared_dataloader = accelerator.prepare(dataloader)
    assert isinstance(prepared_dataloader, DataLoaderDispatcher)
    if accelerator.is_main_process:
        logger = logging.root.manager.loggerDict['accelerate.accelerator']
        list_handler = ListHandler()
        logger.addHandler(list_handler)
    batches_for_metrics = []
    for batch in prepared_dataloader:
        batches_for_metrics.append(accelerator.gather_for_metrics(batch))
    assert torch.cat(batches_for_metrics).size(0) == 30
    if accelerator.is_main_process:
        assert len(list_handler.logs) == 0
        logger.removeHandler(list_handler)