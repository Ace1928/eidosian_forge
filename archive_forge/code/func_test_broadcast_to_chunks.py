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
def test_broadcast_to_chunks():
    x = np.random.default_rng().integers(10, size=(5, 1, 6))
    a = from_array(x, chunks=(3, 1, 3))
    for shape, chunks, expected_chunks in [((5, 3, 6), (3, -1, 3), ((3, 2), (3,), (3, 3))), ((5, 3, 6), (3, 1, 3), ((3, 2), (1, 1, 1), (3, 3))), ((2, 5, 3, 6), (1, 3, 1, 3), ((1, 1), (3, 2), (1, 1, 1), (3, 3)))]:
        xb = np.broadcast_to(x, shape)
        ab = broadcast_to(a, shape, chunks=chunks)
        assert_eq(xb, ab)
        assert ab.chunks == expected_chunks
    with pytest.raises(ValueError):
        broadcast_to(a, a.shape, chunks=((2, 3), (1,), (3, 3)))
    with pytest.raises(ValueError):
        broadcast_to(a, a.shape, chunks=((3, 2), (3,), (3, 3)))
    with pytest.raises(ValueError):
        broadcast_to(a, (5, 2, 6), chunks=((3, 2), (3,), (3, 3)))