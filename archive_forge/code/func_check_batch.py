import random
import pytest
import torch
from llama_recipes.data.sampler import LengthBasedBatchSampler
from llama_recipes.data.sampler import DistributedLengthBasedBatchSampler
def check_batch(batch):
    return all(batch) or not any(batch)