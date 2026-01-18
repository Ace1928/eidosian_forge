import ctypes
import logging
import os
import platform
import shutil
import stat
import sys
import tempfile
import subprocess
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.common.envvar as envvar
from pyomo.common.log import LoggingIntercept
from pyomo.common.fileutils import (
from pyomo.common.download import FileDownloader
def test_find_library_user(self):
    self.tmpdir = os.path.abspath(tempfile.mkdtemp())
    os.chdir(self.tmpdir)
    envvar.PYOMO_CONFIG_DIR = self.tmpdir
    config_libdir = os.path.join(self.tmpdir, 'lib')
    os.mkdir(config_libdir)
    config_bindir = os.path.join(self.tmpdir, 'bin')
    os.mkdir(config_bindir)
    ldlibdir_name = 'in_ld_lib'
    ldlibdir = os.path.join(self.tmpdir, ldlibdir_name)
    os.mkdir(ldlibdir)
    os.environ['LD_LIBRARY_PATH'] = os.pathsep + ldlibdir + os.pathsep
    pathdir_name = 'in_path'
    pathdir = os.path.join(self.tmpdir, pathdir_name)
    os.mkdir(pathdir)
    os.environ['PATH'] = os.pathsep + pathdir + os.pathsep
    libExt = _libExt[_system()][0]
    f_in_cwd_ldlib_path = 'f_in_cwd_ldlib_path'
    open(os.path.join(self.tmpdir, f_in_cwd_ldlib_path), 'w').close()
    open(os.path.join(ldlibdir, f_in_cwd_ldlib_path), 'w').close()
    open(os.path.join(pathdir, f_in_cwd_ldlib_path), 'w').close()
    f_in_ldlib_extension = 'f_in_ldlib_extension'
    open(os.path.join(ldlibdir, f_in_ldlib_extension + libExt), 'w').close()
    f_in_path = 'f_in_path'
    open(os.path.join(pathdir, f_in_path), 'w').close()
    f_in_configlib = 'f_in_configlib'
    open(os.path.join(config_libdir, f_in_configlib), 'w').close()
    f_in_configbin = 'f_in_configbin'
    open(os.path.join(config_bindir, f_in_ldlib_extension), 'w').close()
    open(os.path.join(config_bindir, f_in_configbin), 'w').close()
    self._check_file(find_library(f_in_cwd_ldlib_path), os.path.join(self.tmpdir, f_in_cwd_ldlib_path))
    self._check_file(os.path.join(ldlibdir, f_in_cwd_ldlib_path), find_library(f_in_cwd_ldlib_path, cwd=False))
    self._check_file(os.path.join(ldlibdir, f_in_ldlib_extension) + libExt, find_library(f_in_ldlib_extension))
    self._check_file(os.path.join(pathdir, f_in_path), find_library(f_in_path))
    if _system() == 'windows':
        self._check_file(os.path.join(pathdir, f_in_path), find_library(f_in_path, include_PATH=False))
    else:
        self.assertIsNone(find_library(f_in_path, include_PATH=False))
    self._check_file(os.path.join(pathdir, f_in_path), find_library(f_in_path, pathlist=os.pathsep + pathdir + os.pathsep))
    self._check_file(os.path.join(pathdir, f_in_cwd_ldlib_path), find_library(f_in_cwd_ldlib_path, cwd=False, pathlist=[pathdir]))
    self._check_file(os.path.join(config_libdir, f_in_configlib), find_library(f_in_configlib))
    self._check_file(os.path.join(config_bindir, f_in_configbin), find_library(f_in_configbin))
    self.assertIsNone(find_library(f_in_configbin, include_PATH=False))
    self.assertIsNone(find_library(f_in_configlib, pathlist=pathdir))
    self.assertIsNone(find_library(f_in_configbin, pathlist=pathdir))