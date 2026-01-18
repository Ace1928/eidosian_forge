from __future__ import annotations
import gc
import math
import os
import random
import warnings
import weakref
from bz2 import BZ2File
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from gzip import GzipFile
from itertools import repeat
import partd
import pytest
from tlz import groupby, identity, join, merge, pluck, unique, valmap
import dask
import dask.bag as db
from dask.bag.core import (
from dask.bag.utils import assert_eq
from dask.blockwise import Blockwise
from dask.delayed import Delayed
from dask.typing import Graph
from dask.utils import filetexts, tmpdir, tmpfile
from dask.utils_test import add, hlg_layer, hlg_layer_topological, inc
def test_bag_map():
    b = db.from_sequence(range(100), npartitions=10)
    b2 = db.from_sequence(range(100, 200), npartitions=10)
    x = b.compute()
    x2 = b2.compute()

    def myadd(a=1, b=2, c=3):
        return a + b + c
    assert_eq(db.map(myadd, b), list(map(myadd, x)))
    assert_eq(db.map(myadd, a=b), list(map(myadd, x)))
    assert_eq(db.map(myadd, b, b2), list(map(myadd, x, x2)))
    assert_eq(db.map(myadd, b, 10), [myadd(i, 10) for i in x])
    assert_eq(db.map(myadd, 10, b=b), [myadd(10, b=i) for i in x])
    sol = [myadd(i, b=j, c=100) for i, j in zip(x, x2)]
    assert_eq(db.map(myadd, b, b=b2, c=100), sol)
    sol = [myadd(i, c=100) for i, j in zip(x, x2)]
    assert_eq(db.map(myadd, b, c=100), sol)
    x_sum = sum(x)
    sol = [myadd(x_sum, b=i, c=100) for i in x2]
    assert_eq(db.map(myadd, b.sum(), b=b2, c=100), sol)
    sol = [myadd(i, b=x_sum, c=100) for i in x2]
    assert_eq(db.map(myadd, b2, b.sum(), c=100), sol)
    sol = [myadd(a=100, b=x_sum, c=i) for i in x2]
    assert_eq(db.map(myadd, a=100, b=b.sum(), c=b2), sol)
    a = dask.delayed(10)
    assert_eq(db.map(myadd, b, a), [myadd(i, 10) for i in x])
    assert_eq(db.map(myadd, b, b=a), [myadd(i, b=10) for i in x])
    fewer_parts = db.from_sequence(range(100), npartitions=5)
    with pytest.raises(ValueError):
        db.map(myadd, b, fewer_parts)
    with pytest.raises(ValueError):
        db.map(myadd, b.sum(), 1, 2)
    unequal = db.from_sequence(range(110), npartitions=10)
    with pytest.raises(ValueError):
        db.map(myadd, b, unequal, c=b2).compute()
    with pytest.raises(ValueError):
        db.map(myadd, b, b=unequal, c=b2).compute()