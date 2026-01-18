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
def test_delayed_function_attributes_forwarded():

    @delayed
    def add(x, y):
        """This is a docstring"""
        return x + y
    assert add.__name__ == 'add'
    assert add.__doc__ == 'This is a docstring'
    assert add.__wrapped__(1, 2) == 3