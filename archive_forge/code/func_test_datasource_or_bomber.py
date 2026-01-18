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
def test_datasource_or_bomber(with_nimd_env):
    pkg_def = dict(relpath='pkg')
    with TemporaryDirectory() as tmpdir:
        nibd.get_data_path = lambda: [tmpdir]
        ds = datasource_or_bomber(pkg_def)
        with pytest.raises(DataError):
            ds.get_filename('some_file.txt')
        pkg_dir = pjoin(tmpdir, 'pkg')
        os.mkdir(pkg_dir)
        tmpfile = pjoin(pkg_dir, 'config.ini')
        with open(tmpfile, 'w') as fobj:
            fobj.write('[DEFAULT]\n')
            fobj.write('version = 0.2\n')
        ds = datasource_or_bomber(pkg_def)
        ds.get_filename('some_file.txt')
        pkg_def['min version'] = '0.2'
        ds = datasource_or_bomber(pkg_def)
        ds.get_filename('some_file.txt')
        pkg_def['min version'] = '0.3'
        ds = datasource_or_bomber(pkg_def)
        with pytest.raises(DataError):
            ds.get_filename('some_file.txt')