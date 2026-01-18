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
def test_inline_singleton_lists():
    inp = {'b': (list, 'a'), 'c': (f, 'b', 1)}
    out = {'c': (f, (list, 'a'), 1)}
    assert inline_singleton_lists(inp, ['c']) == out
    out = {'c': (f, 'a', 1)}
    assert optimize(inp, ['c'], rename_fused_keys=False) == out
    assert inline_singleton_lists(inp, ['b', 'c']) == inp
    assert optimize(inp, ['b', 'c'], rename_fused_keys=False) == inp
    inp = {'b': (list, 'a'), 'c': (f, 'b', 1), 'd': (f, 'b', 2)}
    assert inline_singleton_lists(inp, ['c', 'd']) == inp
    inp = {'b': (4, 5), 'c': (f, 'b')}
    assert inline_singleton_lists(inp, ['c']) == inp