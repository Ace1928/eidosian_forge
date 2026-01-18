from __future__ import annotations
import math
import random as rnd
import pytest
import dask.bag as db
from dask.bag import random
def test_choices_empty_partition():
    seq = range(10)
    sut = db.from_sequence(seq, partition_size=9)
    sut = sut.repartition(3)
    li = list(random.choices(sut, k=2).compute())
    assert sut.map_partitions(len).compute() == (9, 0, 1)
    assert len(li) == 2
    assert all((i in seq for i in li))