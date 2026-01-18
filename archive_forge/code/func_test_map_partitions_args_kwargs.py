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
def test_map_partitions_args_kwargs():
    x = [random.randint(-100, 100) for i in range(100)]
    y = [random.randint(-100, 100) for i in range(100)]
    dx = db.from_sequence(x, npartitions=10)
    dy = db.from_sequence(y, npartitions=10)

    def maximum(x, y=0):
        y = repeat(y) if isinstance(y, int) else y
        return [max(a, b) for a, b in zip(x, y)]
    sol = maximum(x, y=10)
    assert_eq(db.map_partitions(maximum, dx, y=10), sol)
    assert_eq(dx.map_partitions(maximum, y=10), sol)
    assert_eq(dx.map_partitions(maximum, 10), sol)
    sol = maximum(x, y)
    assert_eq(db.map_partitions(maximum, dx, dy), sol)
    assert_eq(dx.map_partitions(maximum, y=dy), sol)
    assert_eq(dx.map_partitions(maximum, dy), sol)
    dy_mean = dy.mean().apply(int)
    sol = maximum(x, int(sum(y) / len(y)))
    assert_eq(dx.map_partitions(maximum, y=dy_mean), sol)
    assert_eq(dx.map_partitions(maximum, dy_mean), sol)
    dy_mean = dask.delayed(dy_mean)
    assert_eq(dx.map_partitions(maximum, y=dy_mean), sol)
    assert_eq(dx.map_partitions(maximum, dy_mean), sol)