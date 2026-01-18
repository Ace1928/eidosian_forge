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
def test_compute_no_opt():
    from dask.callbacks import Callback
    b = db.from_sequence(range(100), npartitions=4)
    add1 = partial(add, 1)
    mul2 = partial(mul, 2)
    o = b.map(add1).map(mul2)
    keys = []
    with Callback(pretask=lambda key, *args: keys.append(key)):
        o.compute(scheduler='single-threaded', optimize_graph=False)
    assert len([k for k in keys if 'mul' in k[0]]) == 4
    assert len([k for k in keys if 'add' in k[0]]) == 4
    keys = []
    with Callback(pretask=lambda key, *args: keys.append(key)):
        o.compute(scheduler='single-threaded')
    assert len([k for k in keys if 'mul' in k[0]]) == 8
    assert len([k for k in keys if 'add' in k[0]]) == 4
    assert len([k for k in keys if 'add-mul' in k[0]]) == 4