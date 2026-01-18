from __future__ import annotations
import datetime
import functools
import operator
import pickle
from array import array
import pytest
from tlz import curry
from dask import get
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable
from dask.utils import (
from dask.utils_test import inc
@pytest.mark.parametrize('n,expect', [(0, '0 B'), (920, '920 B'), (930, '0.91 kiB'), (921.23 * 2 ** 10, '921.23 kiB'), (931.23 * 2 ** 10, '0.91 MiB'), (921.23 * 2 ** 20, '921.23 MiB'), (931.23 * 2 ** 20, '0.91 GiB'), (921.23 * 2 ** 30, '921.23 GiB'), (931.23 * 2 ** 30, '0.91 TiB'), (921.23 * 2 ** 40, '921.23 TiB'), (931.23 * 2 ** 40, '0.91 PiB'), (2 ** 60, '1024.00 PiB')])
def test_format_bytes(n, expect):
    assert format_bytes(int(n)) == expect