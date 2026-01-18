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
def test_deprecations_on_env_variables(monkeypatch):
    d = {}
    monkeypatch.setenv('DASK_FUSE_AVE_WIDTH', '123')
    with pytest.warns(FutureWarning) as info:
        dask.config.refresh(config=d)
    assert 'optimization.fuse.ave-width' in str(info[0].message)
    assert get('optimization.fuse.ave-width', config=d) == 123