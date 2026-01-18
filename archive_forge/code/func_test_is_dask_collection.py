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
def test_is_dask_collection():

    class DummyCollection:

        def __init__(self, dsk):
            self.dask = dsk

        def __dask_graph__(self):
            return self.dask
    x = delayed(1) + 2
    assert is_dask_collection(x)
    assert not is_dask_collection(2)
    assert is_dask_collection(DummyCollection({}))
    assert not is_dask_collection(DummyCollection)