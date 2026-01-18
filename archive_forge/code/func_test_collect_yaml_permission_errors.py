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
@pytest.mark.skipif(sys.platform == 'win32', reason="Can't make writeonly file on windows")
@pytest.mark.parametrize('kind', ['directory', 'file'])
def test_collect_yaml_permission_errors(tmp_path, kind):
    a = {'x': 1, 'y': 2}
    b = {'y': 3, 'z': 4}
    pa, pb = (tmp_path / 'a.yaml', tmp_path / 'b.yaml')
    pa.write_text(yaml.dump(a))
    pb.write_text(yaml.dump(b))
    if kind == 'directory':
        cant_read = tmp_path
        expected = {}
    else:
        cant_read = pa
        expected = b
    with no_read_permissions(cant_read):
        config = merge(*collect_yaml(paths=[tmp_path]))
        assert config == expected