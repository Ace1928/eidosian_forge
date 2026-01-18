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
@pytest.mark.parametrize('args,kwargs', [((), {'fuse_ave_width': 123}), (({'fuse_ave_width': 123},), {}), (({'fuse-ave-width': 123},), {})])
def test_deprecations_on_set(args, kwargs):
    with pytest.warns(FutureWarning) as info:
        with dask.config.set(*args, **kwargs):
            assert dask.config.get('optimization.fuse.ave-width') == 123
    assert 'optimization.fuse.ave-width' in str(info[0].message)