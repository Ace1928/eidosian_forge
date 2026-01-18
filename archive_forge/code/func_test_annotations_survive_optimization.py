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
def test_annotations_survive_optimization():
    with dask.annotate(foo='bar'):
        graph = HighLevelGraph.from_collections('b', {'a': 1, 'b': (inc, 'a'), 'c': (inc, 'b')}, [])
        d = Delayed('b', graph)
    assert type(d.dask) is HighLevelGraph
    assert len(d.dask.layers) == 1
    assert len(d.dask.layers['b']) == 3
    assert d.dask.layers['b'].annotations == {'foo': 'bar'}
    d_opt, = dask.optimize(d)
    assert type(d_opt.dask) is HighLevelGraph
    assert len(d_opt.dask.layers) == 1
    assert len(d_opt.dask.layers['b']) == 2
    assert d_opt.dask.layers['b'].annotations == {'foo': 'bar'}