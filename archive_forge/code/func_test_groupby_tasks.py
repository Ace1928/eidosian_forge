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
def test_groupby_tasks():
    b = db.from_sequence(range(160), npartitions=4)
    out = b.groupby(lambda x: x % 10, max_branch=4, shuffle='tasks')
    partitions = dask.get(out.dask, out.__dask_keys__())
    for a in partitions:
        for b in partitions:
            if a is not b:
                assert not set(pluck(0, a)) & set(pluck(0, b))
    b = db.from_sequence(range(1000), npartitions=100)
    out = b.groupby(lambda x: x % 123, shuffle='tasks')
    assert len(out.dask) < 100 ** 2
    partitions = dask.get(out.dask, out.__dask_keys__())
    for a in partitions:
        for b in partitions:
            if a is not b:
                assert not set(pluck(0, a)) & set(pluck(0, b))
    b = db.from_sequence(range(10000), npartitions=345)
    out = b.groupby(lambda x: x % 2834, max_branch=24, shuffle='tasks')
    partitions = dask.get(out.dask, out.__dask_keys__())
    for a in partitions:
        for b in partitions:
            if a is not b:
                assert not set(pluck(0, a)) & set(pluck(0, b))