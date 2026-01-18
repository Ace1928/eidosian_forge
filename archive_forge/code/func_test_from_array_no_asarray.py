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
@pytest.mark.parametrize('asarray,cls', [(True, np.ndarray), (False, np.matrix)])
@pytest.mark.parametrize('inline_array', [True, False])
@pytest.mark.filterwarnings('ignore:the matrix subclass')
def test_from_array_no_asarray(asarray, cls, inline_array):

    def assert_chunks_are_of_type(x):
        chunks = compute_as_if_collection(Array, x.dask, x.__dask_keys__())
        for c in concat(chunks) if isinstance(chunks[0], tuple) else chunks:
            assert type(c) is cls
    x = np.matrix(np.arange(100).reshape((10, 10)))
    dx = da.from_array(x, chunks=(5, 5), asarray=asarray, inline_array=inline_array)
    assert_chunks_are_of_type(dx)
    assert_chunks_are_of_type(dx[0:5])
    assert_chunks_are_of_type(dx[0:5][:, 0])