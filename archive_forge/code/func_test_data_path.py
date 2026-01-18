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
def test_data_path(with_nimd_env):
    if DATA_KEY in env:
        del env[DATA_KEY]
    if USER_KEY in env:
        del os.environ[USER_KEY]
    fake_user_dir = '/user/path'
    nibd.get_nipy_system_dir = lambda: '/unlikely/path'
    nibd.get_nipy_user_dir = lambda: fake_user_dir
    old_pth = get_data_path()
    def_dirs = [pjoin(sys.prefix, 'share', 'nipy')]
    if sys.prefix == '/usr':
        def_dirs.append(pjoin('/usr/local', 'share', 'nipy'))
    assert old_pth == def_dirs + ['/user/path']
    tst_pth = '/a/path' + os.path.pathsep + '/b/ path'
    tst_list = ['/a/path', '/b/ path']
    os.environ[DATA_KEY] = tst_list[0]
    assert get_data_path() == tst_list[:1] + old_pth
    os.environ[DATA_KEY] = tst_pth
    assert get_data_path() == tst_list + old_pth
    del os.environ[DATA_KEY]
    with TemporaryDirectory() as tmpdir:
        tmpfile = pjoin(tmpdir, 'config.ini')
        with open(tmpfile, 'w') as fobj:
            fobj.write('[DATA]\n')
            fobj.write(f'path = {tst_pth}')
        nibd.get_nipy_user_dir = lambda: tmpdir
        assert get_data_path() == tst_list + def_dirs + [tmpdir]
    nibd.get_nipy_user_dir = lambda: fake_user_dir
    assert get_data_path() == old_pth
    with TemporaryDirectory() as tmpdir:
        nibd.get_nipy_system_dir = lambda: tmpdir
        tmpfile = pjoin(tmpdir, 'an_example.ini')
        with open(tmpfile, 'w') as fobj:
            fobj.write('[DATA]\n')
            fobj.write(f'path = {tst_pth}\n')
        tmpfile = pjoin(tmpdir, 'another_example.ini')
        with open(tmpfile, 'w') as fobj:
            fobj.write('[DATA]\n')
            fobj.write('path = %s\n' % '/path/two')
        assert get_data_path() == tst_list + ['/path/two'] + old_pth