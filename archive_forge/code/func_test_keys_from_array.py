from __future__ import annotations
import pickle
import types
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from operator import add, setitem
from random import random
from typing import NamedTuple
import cloudpickle
import pytest
from tlz import merge
import dask
import dask.bag as db
from dask import compute
from dask.delayed import Delayed, delayed, to_task_dask
from dask.highlevelgraph import HighLevelGraph
from dask.utils_test import inc
def test_keys_from_array():
    da = pytest.importorskip('dask.array')
    from dask.array.utils import _check_dsk
    X = da.ones((10, 10), chunks=5).to_delayed().flatten()
    xs = [delayed(inc)(x) for x in X]
    _check_dsk(xs[0].dask)