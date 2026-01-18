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
def test_callable_obj():

    class Foo:

        def __init__(self, a):
            self.a = a

        def __call__(self):
            return 2
    foo = Foo(1)
    f = delayed(foo)
    assert f.compute() is foo
    assert f.a.compute() == 1
    assert f().compute() == 2