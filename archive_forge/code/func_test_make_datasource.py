import os
import sys
import tempfile
from os import environ as env
from os.path import join as pjoin
from tempfile import TemporaryDirectory
import pytest
from .. import data as nibd
from ..data import (
from .test_environment import DATA_KEY, USER_KEY, with_environment
def test_make_datasource(with_nimd_env):
    pkg_def = dict(relpath='pkg')
    with TemporaryDirectory() as tmpdir:
        nibd.get_data_path = lambda: [tmpdir]
        with pytest.raises(DataError):
            make_datasource(pkg_def)
        pkg_dir = pjoin(tmpdir, 'pkg')
        os.mkdir(pkg_dir)
        with pytest.raises(DataError):
            make_datasource(pkg_def)
        tmpfile = pjoin(pkg_dir, 'config.ini')
        with open(tmpfile, 'w') as fobj:
            fobj.write('[DEFAULT]\n')
            fobj.write('version = 0.1\n')
        ds = make_datasource(pkg_def, data_path=[tmpdir])
        assert ds.version == '0.1'