from functools import partial
import pytest
from sklearn.datasets.tests.test_common import (
def test_fetch_kddcup99_shuffle(fetch_kddcup99_fxt):
    dataset = fetch_kddcup99_fxt(random_state=0, subset='SA', percent10=True)
    dataset_shuffled = fetch_kddcup99_fxt(random_state=0, subset='SA', shuffle=True, percent10=True)
    assert set(dataset['target']) == set(dataset_shuffled['target'])
    assert dataset_shuffled.data.shape == dataset.data.shape
    assert dataset_shuffled.target.shape == dataset.target.shape