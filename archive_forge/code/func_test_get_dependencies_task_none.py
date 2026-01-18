from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_get_dependencies_task_none():
    dsk = {'foo': None}
    assert get_dependencies(dsk, task=dsk['foo']) == set()