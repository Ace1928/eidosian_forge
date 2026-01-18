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
def test_chunk_non_array_like():
    array = da.from_array([1] + [2, 2] + [3, 3, 3], chunks=((1, 2, 3),))
    out_chunks = ((1, 1, 1),)
    out = array.map_blocks(lambda x: 5, chunks=out_chunks, enforce_ndim=True)
    with pytest.raises(ValueError, match='Dimension mismatch:'):
        out.compute()
    expected = np.array([5, 5, 5])
    out = array.map_blocks(lambda x: 5, chunks=out_chunks)
    try:
        assert_eq(out, expected)
    except AssertionError:
        assert_eq(out, expected, check_chunks=False)
    else:
        raise AssertionError('Expected a ValueError: Dimension mismatch')