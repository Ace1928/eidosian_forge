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
def test_map_blocks2():
    x = np.arange(10, dtype='i8')
    d = from_array(x, chunks=(2,))

    def func(block, block_id=None, c=0):
        return np.ones_like(block) * sum(block_id) + c
    out = d.map_blocks(func, dtype='i8')
    expected = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype='i8')
    assert_eq(out, expected)
    assert same_keys(d.map_blocks(func, dtype='i8'), out)
    out = d.map_blocks(func, dtype='i8', c=1)
    expected = expected + 1
    assert_eq(out, expected)
    assert same_keys(d.map_blocks(func, dtype='i8', c=1), out)