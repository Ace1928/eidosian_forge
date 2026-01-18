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
def test_fuse_config():
    with dask.config.set({'optimization.fuse.active': False}):
        d = {'a': 1, 'b': (inc, 'a')}
        dependencies = {'b': ('a',)}
        assert fuse(d, 'b', dependencies=dependencies) == (d, dependencies)