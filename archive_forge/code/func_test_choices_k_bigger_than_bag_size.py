from __future__ import annotations
import math
import random as rnd
import pytest
import dask.bag as db
from dask.bag import random
def test_choices_k_bigger_than_bag_size():
    seq = range(3)
    sut = db.from_sequence(seq, npartitions=3)
    li = list(random.choices(sut, k=4).compute())
    assert len(li) == 4
    assert all((i in seq for i in li))