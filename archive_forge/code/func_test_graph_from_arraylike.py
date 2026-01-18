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
@pytest.mark.parametrize('inline_array', [True, False])
def test_graph_from_arraylike(inline_array):
    d = 2
    chunk = (2, 3)
    shape = tuple((d * n for n in chunk))
    arr = np.ones(shape)
    dsk = graph_from_arraylike(arr, chunk, shape=shape, name='X', inline_array=inline_array)
    assert isinstance(dsk, HighLevelGraph)
    if inline_array:
        assert len(dsk.layers) == 1
        assert isinstance(hlg_layer_topological(dsk, 0), Blockwise)
    else:
        assert len(dsk.layers) == 2
        assert isinstance(hlg_layer_topological(dsk, 0), MaterializedLayer)
        assert isinstance(hlg_layer_topological(dsk, 1), Blockwise)
    dsk = dict(dsk)
    assert any((arr is v for v in dsk.values())) is not inline_array