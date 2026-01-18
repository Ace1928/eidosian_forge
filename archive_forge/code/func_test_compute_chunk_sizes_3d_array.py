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
def test_compute_chunk_sizes_3d_array(N=8):
    X = np.linspace(-1, 2, num=8 * 8 * 8).reshape(8, 8, 8)
    X = da.from_array(X, chunks=(4, 4, 4))
    idx = X.sum(axis=0).sum(axis=0) > 0
    Y = X[idx]
    idx = X.sum(axis=1).sum(axis=1) < 0
    Y = Y[:, idx]
    idx = X.sum(axis=2).sum(axis=1) > 0.1
    Y = Y[:, :, idx]
    assert Y.compute().shape == (8, 3, 5)
    assert X.compute().shape == (8, 8, 8)
    assert Y.chunks == ((np.nan, np.nan),) * 3
    assert all((np.isnan(s) for s in Y.shape))
    Z = Y.compute_chunk_sizes()
    assert Z is Y
    assert Z.shape == (8, 3, 5)
    assert Z.chunks == ((4, 4), (3, 0), (1, 4))