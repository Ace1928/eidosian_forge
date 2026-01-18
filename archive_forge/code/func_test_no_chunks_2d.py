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
def test_no_chunks_2d():
    X = np.arange(24).reshape((4, 6))
    x = da.from_array(X, chunks=(2, 2))
    x._chunks = ((np.nan, np.nan), (np.nan, np.nan, np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        assert_eq(da.log(x), np.log(X))
    assert_eq(x.T, X.T)
    assert_eq(x.sum(axis=0, keepdims=True), X.sum(axis=0, keepdims=True))
    assert_eq(x.sum(axis=1, keepdims=True), X.sum(axis=1, keepdims=True))
    assert_eq(x.dot(x.T + 1), X.dot(X.T + 1))