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
def test_compute_nested():
    a = delayed(1) + 5
    b = a + 1
    c = a + 2
    assert compute({'a': a, 'b': [1, 2, b]}, (c, 2)) == ({'a': 6, 'b': [1, 2, 7]}, (8, 2))
    res = compute([a, b], c, traverse=False)
    assert res[0][0] is a
    assert res[0][1] is b
    assert res[1] == 8