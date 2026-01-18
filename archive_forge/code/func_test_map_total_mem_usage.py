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
def test_map_total_mem_usage():
    """https://github.com/dask/dask/issues/10338"""
    b = db.from_sequence(range(1, 100), npartitions=3)
    total_mem_b = sum(b.map_partitions(total_mem_usage).compute())
    c = b.map(lambda x: x)
    total_mem_c = sum(c.map_partitions(total_mem_usage).compute())
    assert total_mem_b == total_mem_c