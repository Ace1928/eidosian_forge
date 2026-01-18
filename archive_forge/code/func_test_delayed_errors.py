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
def test_delayed_errors():
    a = delayed([1, 2, 3])
    pytest.raises(TypeError, lambda: setattr(a, 'foo', 1))
    pytest.raises(TypeError, lambda: setitem(a, 1, 0))
    pytest.raises(TypeError, lambda: 1 in a)
    pytest.raises(TypeError, lambda: list(a))
    pytest.raises(AttributeError, lambda: a._hidden())
    pytest.raises(TypeError, lambda: bool(a))