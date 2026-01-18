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
@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skip(reason='Hangs')
def test_from_url():
    a = db.from_url(['http://google.com', 'http://github.com'])
    assert a.npartitions == 2
    b = db.from_url('http://raw.githubusercontent.com/dask/dask/main/README.rst')
    assert b.npartitions == 1
    assert b'Dask\n' in b.take(10)