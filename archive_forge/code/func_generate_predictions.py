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
def generate_predictions(model, dataloader, accelerator):
    logits_and_targets = []
    for batch in dataloader:
        input, target = batch.values()
        with torch.no_grad():
            logit = model(input)
            logit, target = accelerator.gather_for_metrics((logit, target))
            logits_and_targets.append((logit, target))
    logits, targs = ([], [])
    for logit, targ in logits_and_targets:
        logits.append(logit)
        targs.append(targ)
    logits, targs = (torch.cat(logits), torch.cat(targs))
    return (logits, targs)