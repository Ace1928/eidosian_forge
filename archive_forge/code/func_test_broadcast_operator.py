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
@pytest.mark.parametrize('u_shape, v_shape', [[tuple(), (2, 3)], [(1,), (2, 3)], [(1, 1), (2, 3)], [(0, 3), (1, 3)], [(2, 0), (2, 1)], [(1, 0), (2, 1)], [(0, 1), (1, 3)]])
def test_broadcast_operator(u_shape, v_shape):
    rng = np.random.default_rng()
    u = rng.random(u_shape)
    v = rng.random(v_shape)
    d_u = from_array(u, chunks=1)
    d_v = from_array(v, chunks=1)
    w = u * v
    d_w = d_u * d_v
    assert_eq(w, d_w)