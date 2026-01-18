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
@pytest.mark.parametrize('chunks', (100, 6))
@pytest.mark.parametrize('other', [[0, 0, 1], [2, 1, 3], (0, 0, 1)])
def test_elemwise_with_lists(chunks, other):
    x = np.arange(12).reshape((4, 3))
    d = da.arange(12, chunks=chunks).reshape((4, 3))
    x2 = np.vstack([x[:, 0], x[:, 1], x[:, 2]]).T
    d2 = da.vstack([d[:, 0], d[:, 1], d[:, 2]]).T
    assert_eq(x2, d2)
    x3 = x2 * other
    d3 = d2 * other
    assert_eq(x3, d3)