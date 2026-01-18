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
def test_map_releases_element_references_as_soon_as_possible():

    class C:

        def __init__(self, i):
            self.i = i
    in_memory = weakref.WeakSet()

    def f_create(i):
        assert len(in_memory) == 0
        o = C(i)
        in_memory.add(o)
        return o

    def f_drop(o):
        return o.i + 100
    b = db.from_sequence(range(2), npartitions=1).map(f_create).map(f_drop).map(f_create).map(f_drop).sum()
    try:
        gc.disable()
        b.compute(scheduler='sync')
    finally:
        gc.enable()