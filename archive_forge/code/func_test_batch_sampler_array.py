import random
import pytest
import torch
from llama_recipes.data.sampler import LengthBasedBatchSampler
from llama_recipes.data.sampler import DistributedLengthBasedBatchSampler
@pytest.mark.parametrize('batch_size, drop_last', [(2, False), (8, False), (2, True), (8, True)])
def test_batch_sampler_array(dataset, batch_size, drop_last):
    sampler = LengthBasedBatchSampler(dataset, batch_size, drop_last)
    EXPECTED_LENGTH = SAMPLES // batch_size if drop_last else SAMPLES // batch_size + SAMPLES % batch_size
    all_ids = [i for b in sampler for i in b]
    assert len(set(all_ids)) == EXPECTED_LENGTH * batch_size if drop_last else len(dataset)
    assert len(sampler) == EXPECTED_LENGTH
    is_long = [len(d) >= 10 for d in dataset]

    def check_batch(batch):
        return all(batch) or not any(batch)
    assert all((check_batch((is_long[i] for i in b)) for b in sampler))