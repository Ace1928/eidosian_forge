from __future__ import annotations
import itertools
import pickle
from functools import partial
import pytest
import dask
from dask.base import tokenize
from dask.core import get_dependencies
from dask.local import get_sync
from dask.optimization import (
from dask.utils import apply, partial_by_order
from dask.utils_test import add, inc
def inc_only_once(x):
    nonlocal already_called
    if already_called:
        raise RuntimeError
    already_called = True
    return x + 1