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
def test_distinct_with_key():
    seq = [{'a': i} for i in [0, 1, 2, 1, 2, 3, 2, 3, 4, 5]]
    bag = db.from_sequence(seq, npartitions=3)
    expected = list(unique(seq, key=lambda x: x['a']))
    assert_eq(bag.distinct(key='a'), expected)
    assert_eq(bag.distinct(key=lambda x: x['a']), expected)