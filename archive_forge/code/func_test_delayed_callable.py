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
def test_delayed_callable():
    f = delayed(add, pure=True)
    v = f(1, 2)
    assert v.dask == {v.key: (add, 1, 2)}
    assert f.dask == {f.key: add}
    assert f.compute() == add