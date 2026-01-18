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
def test_clone_key():
    assert clone_key('inc-1-2-3', 123) == 'inc-73db79fdf4518507ddc84796726d4844'
    assert clone_key('x', 123) == 'x-c4fb64ccca807af85082413d7ef01721'
    assert clone_key('x', 456) == 'x-d4b538b4d4cf68fca214077609feebae'
    assert clone_key(('x', 1), 456) == ('x-d4b538b4d4cf68fca214077609feebae', 1)
    assert clone_key(('sum-1-2-3', h1, 1), 123) == ('sum-822e7622aa1262cef988b3033c32aa37', h1, 1)
    with pytest.raises(TypeError):
        clone_key(1, 2)