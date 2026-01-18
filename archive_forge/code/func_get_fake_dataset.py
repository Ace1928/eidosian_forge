import pytest
from pytest import approx
from unittest.mock import patch
import torch
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler
from llama_recipes.finetuning import main
from llama_recipes.data.sampler import LengthBasedBatchSampler
def get_fake_dataset():
    return [{'input_ids': [1], 'attention_mask': [1], 'labels': [1]}]