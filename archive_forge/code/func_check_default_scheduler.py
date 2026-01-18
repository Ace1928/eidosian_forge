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
def check_default_scheduler(module, collection, expected, emscripten):
    from contextlib import nullcontext
    from unittest import mock
    from dask.local import get_sync
    if emscripten:
        ctx = mock.patch('dask.base.named_schedulers', {'sync': get_sync})
    else:
        ctx = nullcontext()
    with ctx:
        import importlib
        if expected == 'sync':
            from dask.local import get_sync as get
        elif expected == 'threads':
            from dask.threaded import get
        elif expected == 'processes':
            from dask.multiprocessing import get
        mod = importlib.import_module(module)
        assert getattr(mod, collection).__dask_scheduler__ == get