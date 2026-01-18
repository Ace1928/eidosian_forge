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
def test_fold():
    c = b.fold(add)
    assert c.compute() == sum(L)
    assert c.key == b.fold(add).key
    c2 = b.fold(add, initial=10)
    assert c2.key != c.key
    assert c2.compute() == sum(L) + 10 * b.npartitions
    assert c2.key == b.fold(add, initial=10).key
    c = db.from_sequence(range(5), npartitions=3)

    def binop(acc, x):
        acc = acc.copy()
        acc.add(x)
        return acc
    d = c.fold(binop, set.union, initial=set())
    assert d.compute() == set(c)
    assert d.key == c.fold(binop, set.union, initial=set()).key
    d = db.from_sequence('hello')
    assert set(d.fold(lambda a, b: ''.join([a, b]), initial='').compute()) == set('hello')
    e = db.from_sequence([[1], [2], [3]], npartitions=2)
    assert set(e.fold(add, initial=[]).compute(scheduler='sync')) == {1, 2, 3}