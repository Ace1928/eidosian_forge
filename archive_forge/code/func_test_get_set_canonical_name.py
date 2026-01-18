from __future__ import annotations
import os
import pathlib
import site
import stat
import sys
from collections import OrderedDict
from contextlib import contextmanager
import pytest
import yaml
import dask.config
from dask.config import (
def test_get_set_canonical_name():
    c = {'x-y': {'a_b': 123}}
    keys = ['x_y.a_b', 'x-y.a-b', 'x_y.a-b']
    for k in keys:
        assert dask.config.get(k, config=c) == 123
    with dask.config.set({'x_y': {'a-b': 456}}, config=c):
        for k in keys:
            assert dask.config.get(k, config=c) == 456
    with dask.config.set({'x_y': {'a-b': {'c_d': 1}, 'e-f': 2}}, config=c):
        assert dask.config.get('x_y.a-b', config=c) == {'c_d': 1}
        assert dask.config.get('x_y.e_f', config=c) == 2