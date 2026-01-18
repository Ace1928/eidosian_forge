from __future__ import annotations
import copy
import os
import re
import sys
from functools import partial
from operator import add, neg
import pytest
from dask.dot import _to_cytoscape_json, cytoscape_graph
from dask import delayed
from dask.utils import ensure_not_exists
def test_delayed_kwargs_apply():

    def f(x, y=True):
        return x + y
    x = delayed(f)(1, y=2)
    label = task_label(x.dask[x.key])
    assert 'f' in label
    assert 'apply' not in label