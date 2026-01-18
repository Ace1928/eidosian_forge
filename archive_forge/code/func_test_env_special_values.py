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
@pytest.mark.parametrize('preproc', [lambda x: x, lambda x: x.lower(), lambda x: x.upper()])
@pytest.mark.parametrize('v,out', [('None', None), ('Null', None), ('False', False), ('True', True)])
def test_env_special_values(preproc, v, out):
    env = {'DASK_A': preproc(v)}
    res = collect_env(env)
    assert res == {'a': out}