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
def test_mrpc(dispatch_batches: bool=False, split_batches: bool=False):
    metric = evaluate.load('glue', 'mrpc')
    setup, accelerator = get_mrpc_setup(dispatch_batches, split_batches)
    model, dataloader, device = setup['no']
    model.to(device)
    model.eval()
    for batch in dataloader:
        batch.to(device)
        with torch.inference_mode():
            outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        metric.add_batch(predictions=preds, references=batch['labels'])
    baseline = metric.compute()
    model, dataloader, device = setup['ddp']
    model.eval()
    for batch in dataloader:
        with torch.inference_mode():
            outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        references = batch['labels']
        preds, references = accelerator.gather_for_metrics((preds, references))
        metric.add_batch(predictions=preds, references=references)
    distributed = metric.compute()
    for key in 'accuracy f1'.split():
        assert math.isclose(baseline[key], distributed[key]), f'Baseline and Distributed are not the same for key {key}:\n\tBaseline: {baseline[key]}\n\tDistributed: {distributed[key]}\n'