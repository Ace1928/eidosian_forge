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
def test_asciitable():
    res = asciitable(['fruit', 'color'], [('apple', 'red'), ('banana', 'yellow'), ('tomato', 'red'), ('pear', 'green')])
    assert res == '+--------+--------+\n| fruit  | color  |\n+--------+--------+\n| apple  | red    |\n| banana | yellow |\n| tomato | red    |\n| pear   | green  |\n+--------+--------+'