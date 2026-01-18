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
def test_broadcast_chunks():
    assert broadcast_chunks() == ()
    assert broadcast_chunks(((2, 3),)) == ((2, 3),)
    assert broadcast_chunks(((5, 5),), ((5, 5),)) == ((5, 5),)
    a = ((10, 10, 10), (5, 5))
    b = ((5, 5),)
    assert broadcast_chunks(a, b) == ((10, 10, 10), (5, 5))
    assert broadcast_chunks(b, a) == ((10, 10, 10), (5, 5))
    a = ((10, 10, 10), (5, 5))
    b = ((1,), (5, 5))
    assert broadcast_chunks(a, b) == ((10, 10, 10), (5, 5))
    a = ((10, 10, 10), (5, 5))
    b = ((3, 3), (5, 5))
    with pytest.raises(ValueError):
        broadcast_chunks(a, b)
    a = ((1,), (5, 5))
    b = ((1,), (5, 5))
    assert broadcast_chunks(a, b) == a
    a = ((1,), (np.nan, np.nan, np.nan))
    b = ((3, 3), (1,))
    r = broadcast_chunks(a, b)
    assert r[0] == b[0] and np.allclose(r[1], a[1], equal_nan=True)
    a = ((3, 3), (1,))
    b = ((1,), (np.nan, np.nan, np.nan))
    r = broadcast_chunks(a, b)
    assert r[0] == a[0] and np.allclose(r[1], b[1], equal_nan=True)
    a = ((3, 3), (5, 5))
    b = ((1,), (np.nan, np.nan, np.nan))
    with pytest.raises(ValueError):
        broadcast_chunks(a, b)