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
def test_delayed_decorator_on_method():

    class A:
        BASE = 10

        def __init__(self, base):
            self.BASE = base

        @delayed
        def addmethod(self, x, y):
            return self.BASE + x + y

        @classmethod
        @delayed
        def addclass(cls, x, y):
            return cls.BASE + x + y

        @staticmethod
        @delayed
        def addstatic(x, y):
            return x + y
    a = A(100)
    assert a.addmethod(3, 4).compute() == 107
    assert A.addmethod(a, 3, 4).compute() == 107
    assert a.addclass(3, 4).compute() == 17
    assert A.addclass(3, 4).compute() == 17
    assert a.addstatic(3, 4).compute() == 7
    assert A.addstatic(3, 4).compute() == 7
    assert isinstance(a.addmethod, types.MethodType)
    assert isinstance(A.addclass, types.MethodType)
    assert isinstance(A.addstatic, Delayed)