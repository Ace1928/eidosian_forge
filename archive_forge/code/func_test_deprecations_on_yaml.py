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
@pytest.mark.parametrize('key', ['fuse-ave-width', 'fuse_ave_width'])
def test_deprecations_on_yaml(tmp_path, key):
    d = {}
    (tmp_path / 'dask.yaml').write_text(yaml.dump({key: 123}))
    with pytest.warns(FutureWarning) as info:
        dask.config.refresh(config=d, paths=[tmp_path])
    assert 'optimization.fuse.ave-width' in str(info[0].message)
    assert get('optimization.fuse.ave-width', config=d) == 123