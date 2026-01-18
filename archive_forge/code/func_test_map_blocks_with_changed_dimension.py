from __future__ import annotations
import contextlib
import copy
import pathlib
import re
import xml.etree.ElementTree
from unittest import mock
import pytest
import math
import operator
import os
import time
import warnings
from functools import reduce
from io import StringIO
from operator import add, sub
from threading import Lock
from tlz import concat, merge
from tlz.curried import identity
import dask
import dask.array as da
from dask.array.chunk import getitem
from dask.array.core import (
from dask.array.numpy_compat import NUMPY_GE_200
from dask.array.reshape import _not_implemented_message
from dask.array.tests.test_dispatch import EncapsulateNDArray
from dask.array.utils import assert_eq, same_keys
from dask.base import compute_as_if_collection, tokenize
from dask.blockwise import broadcast_dimensions
from dask.blockwise import make_blockwise_graph as top
from dask.blockwise import optimize_blockwise
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
from dask.layers import Blockwise
from dask.utils import SerializableLock, key_split, parse_bytes, tmpdir, tmpfile
from dask.utils_test import dec, hlg_layer_topological, inc
def test_map_blocks_with_changed_dimension():
    x = np.arange(56).reshape((7, 8))
    d = da.from_array(x, chunks=(7, 4))
    e = d.map_blocks(lambda b: b.sum(axis=0), chunks=(4,), drop_axis=0, dtype=d.dtype)
    assert e.chunks == ((4, 4),)
    assert_eq(e, x.sum(axis=0))
    with pytest.raises(ValueError):
        d.map_blocks(lambda b: b.sum(axis=0), chunks=(), drop_axis=0)
    with pytest.raises(ValueError):
        d.map_blocks(lambda b: b.sum(axis=0), chunks=((4, 4, 4),), drop_axis=0)
    with pytest.raises(ValueError):
        d.map_blocks(lambda b: b.sum(axis=1), chunks=((3, 4),), drop_axis=1)
    d = da.from_array(x, chunks=(4, 8))
    e = d.map_blocks(lambda b: b.sum(axis=1), drop_axis=1, dtype=d.dtype)
    assert e.chunks == ((4, 3),)
    assert_eq(e, x.sum(axis=1))
    x = np.arange(64).reshape((8, 8))
    d = da.from_array(x, chunks=(4, 4))
    e = d.map_blocks(lambda b: b[None, :, :, None], chunks=(1, 4, 4, 1), new_axis=[0, 3], dtype=d.dtype)
    assert e.chunks == ((1,), (4, 4), (4, 4), (1,))
    assert_eq(e, x[None, :, :, None])
    e = d.map_blocks(lambda b: b[None, :, :, None], new_axis=[0, 3], dtype=d.dtype)
    assert e.chunks == ((1,), (4, 4), (4, 4), (1,))
    assert_eq(e, x[None, :, :, None])
    with pytest.raises(ValueError):
        d.map_blocks(lambda b: b, new_axis=(3, 4))
    d = da.from_array(x, chunks=(8, 4))
    e = d.map_blocks(lambda b: b.sum(axis=0)[:, None, None], drop_axis=0, new_axis=(1, 2), dtype=d.dtype)
    assert e.chunks == ((4, 4), (1,), (1,))
    assert_eq(e, x.sum(axis=0)[:, None, None])
    d = da.from_array(x, chunks=(4, 8))
    e = d.map_blocks(lambda b: b.sum(axis=1)[:, None, None], drop_axis=1, new_axis=(1, 2), dtype=d.dtype)
    assert e.chunks == ((4, 4), (1,), (1,))
    assert_eq(e, x.sum(axis=1)[:, None, None])