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
def test_map_method():
    b = db.from_sequence(range(100), npartitions=10)
    b2 = db.from_sequence(range(100, 200), npartitions=10)
    x = b.compute()
    x2 = b2.compute()

    def myadd(a, b=2, c=3):
        return a + b + c
    assert b.map(myadd).compute() == list(map(myadd, x))
    assert b.map(myadd, b2).compute() == list(map(myadd, x, x2))
    assert b.map(myadd, 10).compute() == [myadd(i, 10) for i in x]
    assert b.map(myadd, b=10).compute() == [myadd(i, b=10) for i in x]
    assert b.map(myadd, b2, c=10).compute() == [myadd(i, j, 10) for i, j in zip(x, x2)]
    x_sum = sum(x)
    assert b.map(myadd, b.sum(), c=10).compute() == [myadd(i, x_sum, 10) for i in x]