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
def test_frequencies():
    c = b.frequencies()
    assert dict(c) == {0: 3, 1: 3, 2: 3, 3: 3, 4: 3}
    c2 = b.frequencies(split_every=2)
    assert dict(c2) == {0: 3, 1: 3, 2: 3, 3: 3, 4: 3}
    assert c.name == b.frequencies().name
    assert c.name != c2.name
    assert c2.name == b.frequencies(split_every=2).name
    b2 = db.from_sequence(range(20), partition_size=2)
    b2 = b2.filter(lambda x: x < 10)
    d = b2.frequencies()
    assert dict(d) == dict(zip(range(10), [1] * 10))
    bag = db.from_sequence([0, 0, 0, 0], npartitions=4)
    bag2 = bag.filter(None).frequencies(split_every=2)
    assert_eq(bag2, [])