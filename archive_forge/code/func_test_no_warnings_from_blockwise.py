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
def test_no_warnings_from_blockwise():
    with warnings.catch_warnings(record=True) as record:
        x = da.ones((3, 10, 10), chunks=(3, 2, 2))
        da.map_blocks(lambda y: np.mean(y, axis=0), x, dtype=x.dtype, drop_axis=0)
    assert not record
    with warnings.catch_warnings(record=True) as record:
        x = da.ones((15, 15), chunks=(5, 5))
        (x.dot(x.T + 1) - x.mean(axis=0)).std()
    assert not record
    with warnings.catch_warnings(record=True) as record:
        x = da.ones((1,), chunks=(1,))
        1 / x[0]
    assert not record