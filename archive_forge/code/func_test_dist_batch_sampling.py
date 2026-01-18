import random
import pytest
import torch
from llama_recipes.data.sampler import LengthBasedBatchSampler
from llama_recipes.data.sampler import DistributedLengthBasedBatchSampler
@pytest.mark.parametrize('batch_size', [2, 8])
def test_dist_batch_sampling(dataset, batch_size):
    sampler_1 = DistributedLengthBasedBatchSampler(dataset, batch_size=batch_size, rank=0, num_replicas=2, shuffle=False)
    sampler_2 = DistributedLengthBasedBatchSampler(dataset, batch_size=batch_size, rank=1, num_replicas=2, shuffle=False)
    ids_1 = set((i for b in sampler_1 for i in b))
    ids_2 = set((i for b in sampler_2 for i in b))
    assert ids_1.isdisjoint(ids_2)
    assert len(ids_1) + len(ids_2) > 0
    assert len(ids_1) + len(ids_2) == len(dataset) // batch_size * batch_size