from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_literal_serializable():
    l = literal((add, 1, 2))
    assert pickle.loads(pickle.dumps(l)).data == (add, 1, 2)