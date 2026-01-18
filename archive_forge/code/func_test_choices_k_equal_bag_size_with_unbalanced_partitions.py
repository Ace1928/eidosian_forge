from __future__ import annotations
import math
import random as rnd
import pytest
import dask.bag as db
from dask.bag import random
def test_choices_k_equal_bag_size_with_unbalanced_partitions():
    seq = range(10)
    sut = db.from_sequence(seq, partition_size=9)
    li = list(random.choices(sut, k=10).compute())
    assert sut.map_partitions(len).compute() == (9, 1)
    assert len(li) == 10
    assert all((i in seq for i in li))