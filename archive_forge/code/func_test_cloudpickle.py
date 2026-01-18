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
@pytest.mark.parametrize('f', [delayed(modlevel_eager), modlevel_delayed1, modlevel_delayed2])
def test_cloudpickle(f):
    d = f(2)
    d = cloudpickle.loads(cloudpickle.dumps(d, protocol=pickle.HIGHEST_PROTOCOL))
    assert d.compute() == 3