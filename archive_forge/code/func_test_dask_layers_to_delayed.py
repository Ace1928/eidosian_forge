from __future__ import annotations
import gc
import math
import os
import random
import warnings
import weakref
from bz2 import BZ2File
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from gzip import GzipFile
from itertools import repeat
import partd
import pytest
from tlz import groupby, identity, join, merge, pluck, unique, valmap
import dask
import dask.bag as db
from dask.bag.core import (
from dask.bag.utils import assert_eq
from dask.blockwise import Blockwise
from dask.delayed import Delayed
from dask.typing import Graph
from dask.utils import filetexts, tmpdir, tmpfile
from dask.utils_test import add, hlg_layer, hlg_layer_topological, inc
@pytest.mark.parametrize('optimize', [False, True])
def test_dask_layers_to_delayed(optimize):
    da = pytest.importorskip('dask.array')
    i = db.Item.from_delayed(da.ones(1).to_delayed()[0])
    name = i.key[0]
    assert i.key[1:] == (0,)
    assert i.dask.layers.keys() == {'delayed-' + name}
    assert i.dask.dependencies == {'delayed-' + name: set()}
    assert i.__dask_layers__() == ('delayed-' + name,)
    arr = da.ones(1) + 1
    delayed = arr.to_delayed(optimize_graph=optimize)[0]
    i = db.Item.from_delayed(delayed)
    assert i.key == delayed.key
    assert i.dask is delayed.dask
    assert i.__dask_layers__() == delayed.__dask_layers__()
    back = i.to_delayed(optimize_graph=optimize)
    assert back.__dask_layers__() == i.__dask_layers__()
    if not optimize:
        assert back.dask is arr.dask
        with pytest.raises(ValueError, match='not in'):
            db.Item(back.dask, back.key)
    with pytest.raises(ValueError, match='not in'):
        db.Item(arr.dask, (arr.name,), layer='foo')