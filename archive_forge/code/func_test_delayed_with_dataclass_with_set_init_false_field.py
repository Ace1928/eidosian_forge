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
def test_delayed_with_dataclass_with_set_init_false_field():

    @dataclass
    class ADataClass:
        a: int
        b: int = field(init=False)
    literal = dask.delayed(3)

    def prep_dataclass(a):
        data = ADataClass(a=a)
        data.b = 4
        return data
    with pytest.raises(ValueError) as e:
        dask.delayed(prep_dataclass(literal))
    e.match('ADataClass')
    e.match('`init=False` are not supported')