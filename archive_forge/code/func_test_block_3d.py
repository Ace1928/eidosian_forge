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
def test_block_3d():
    a000 = np.ones((2, 2, 2), int) * 1
    a100 = np.ones((3, 2, 2), int) * 2
    a010 = np.ones((2, 3, 2), int) * 3
    a001 = np.ones((2, 2, 3), int) * 4
    a011 = np.ones((2, 3, 3), int) * 5
    a101 = np.ones((3, 2, 3), int) * 6
    a110 = np.ones((3, 3, 2), int) * 7
    a111 = np.ones((3, 3, 3), int) * 8
    d000 = da.asarray(a000)
    d100 = da.asarray(a100)
    d010 = da.asarray(a010)
    d001 = da.asarray(a001)
    d011 = da.asarray(a011)
    d101 = da.asarray(a101)
    d110 = da.asarray(a110)
    d111 = da.asarray(a111)
    expected = np.block([[[a000, a001], [a010, a011]], [[a100, a101], [a110, a111]]])
    result = da.block([[[d000, d001], [d010, d011]], [[d100, d101], [d110, d111]]])
    assert_eq(expected, result)
    expected = np.block([[[a000, a001[:, :, :0]], [a010[:, :0, :], a011[:, :0, :0]]], [[a100[:0, :, :], a101[:0, :, :0]], [a110[:0, :0, :], a111[:0, :0, :0]]]])
    result = da.block([[[d000, d001[:, :, :0]], [d010[:, :0, :], d011[:, :0, :0]]], [[d100[:0, :, :], d101[:0, :, :0]], [d110[:0, :0, :], d111[:0, :0, :0]]]])
    assert result is d000
    assert_eq(expected, result)