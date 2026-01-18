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
def test_accumulate():
    parts = [[1, 2, 3], [4, 5], [], [6, 7]]
    dsk = {('test', i): p for i, p in enumerate(parts)}
    b = db.Bag(dsk, 'test', len(parts))
    r = b.accumulate(add)
    assert r.name == b.accumulate(add).name
    assert r.name != b.accumulate(add, -1).name
    assert r.compute() == [1, 3, 6, 10, 15, 21, 28]
    assert b.accumulate(add, -1).compute() == [-1, 0, 2, 5, 9, 14, 20, 27]
    assert b.accumulate(add).map(inc).compute() == [2, 4, 7, 11, 16, 22, 29]
    b = db.from_sequence([1, 2, 3], npartitions=1)
    assert b.accumulate(add).compute() == [1, 3, 6]