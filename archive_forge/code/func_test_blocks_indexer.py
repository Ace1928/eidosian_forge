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
def test_blocks_indexer():
    x = da.arange(10, chunks=2)
    assert isinstance(x.blocks[0], da.Array)
    assert_eq(x.blocks[0], x[:2])
    assert_eq(x.blocks[-1], x[-2:])
    assert_eq(x.blocks[:3], x[:6])
    assert_eq(x.blocks[[0, 1, 2]], x[:6])
    assert_eq(x.blocks[[3, 0, 2]], np.array([6, 7, 0, 1, 4, 5]))
    x = da.random.default_rng().random((20, 20), chunks=(4, 5))
    assert_eq(x.blocks[0], x[:4])
    assert_eq(x.blocks[0, :3], x[:4, :15])
    assert_eq(x.blocks[:, :3], x[:, :15])
    x = da.ones((40, 40, 40), chunks=(10, 10, 10))
    assert_eq(x.blocks[0, :, 0], np.ones((10, 40, 10)))
    x = da.ones((2, 2), chunks=1)
    with pytest.raises(ValueError):
        x.blocks[[0, 1], [0, 1]]
    with pytest.raises(ValueError):
        x.blocks[np.array([0, 1]), [0, 1]]
    with pytest.raises(ValueError) as info:
        x.blocks[np.array([0, 1]), np.array([0, 1])]
    assert 'list' in str(info.value)
    with pytest.raises(ValueError) as info:
        x.blocks[None, :, :]
    assert 'newaxis' in str(info.value) and 'not supported' in str(info.value)
    with pytest.raises(IndexError) as info:
        x.blocks[100, 100]