from __future__ import annotations
import dataclasses
import inspect
import os
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import Executor
from operator import add, mul
from typing import NamedTuple
import pytest
from tlz import merge, partial
import dask
import dask.bag as db
from dask.base import (
from dask.delayed import Delayed, delayed
from dask.diagnostics import Profiler
from dask.highlevelgraph import HighLevelGraph
from dask.utils import tmpdir, tmpfile
from dask.utils_test import dec, import_or_none, inc
def test_get_collection_names():

    class DummyCollection:

        def __init__(self, dsk, keys):
            self.dask = dsk
            self.keys = keys

        def __dask_graph__(self):
            return self.dask

        def __dask_keys__(self):
            return self.keys
    with pytest.raises(TypeError):
        get_collection_names(object())
    with pytest.raises(TypeError):
        get_collection_names(DummyCollection({1: 2}, [1]))
    with pytest.raises(TypeError):
        get_collection_names(DummyCollection({(): 1}, [()]))
    with pytest.raises(TypeError):
        get_collection_names(DummyCollection({(1,): 1}, [(1,)]))
    assert get_collection_names(DummyCollection({}, [])) == set()
    assert get_collection_names(DummyCollection({('a-1', h1): 1, ('a-1', h2): 2, 'b-2': 3, 'c': 4}, [[[('a-1', h1), ('a-1', h2), 'b-2', 'c']]])) == {'a-1', 'b-2', 'c'}