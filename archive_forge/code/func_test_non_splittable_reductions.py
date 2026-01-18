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
@pytest.mark.parametrize('npartitions', [1, 10])
def test_non_splittable_reductions(npartitions):
    np = pytest.importorskip('numpy')
    data = list(range(100))
    c = db.from_sequence(data, npartitions=npartitions)
    assert_eq(c.mean(), np.mean(data))
    assert_eq(c.std(), np.std(data))