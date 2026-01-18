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
def test_findfile(self):
    self.tmpdir = os.path.abspath(tempfile.mkdtemp())
    subdir_name = 'aaa'
    subdir = os.path.join(self.tmpdir, subdir_name)
    os.mkdir(subdir)
    os.chdir(self.tmpdir)
    fname = 'foo.py'
    self.assertEqual(None, find_file(fname))
    open(os.path.join(self.tmpdir, fname), 'w').close()
    open(os.path.join(subdir, fname), 'w').close()
    open(os.path.join(subdir, 'aaa'), 'w').close()
    self._check_file(find_file(fname), os.path.join(self.tmpdir, fname))
    self.assertIsNone(find_file(fname, cwd=False))
    self._check_file(find_file(fname, pathlist=[subdir]), os.path.join(self.tmpdir, fname))
    self._check_file(find_file(fname, pathlist=[subdir], cwd=False), os.path.join(subdir, fname))
    found = find_file(fname, pathlist=[subdir], mode=os.X_OK)
    if _system() in ('windows', 'cygwin'):
        self._check_file(found, os.path.join(self.tmpdir, fname))
    else:
        self.assertIsNone(found)
    self._make_exec(os.path.join(subdir, fname))
    found = find_file(fname, pathlist=[subdir], mode=os.X_OK)
    if _system() in ('windows', 'cygwin'):
        ref = os.path.join(self.tmpdir, fname)
    else:
        ref = os.path.join(subdir, fname)
    self._check_file(found, ref)
    self._check_file(find_file(fname, pathlist=os.pathsep + subdir + os.pathsep, cwd=False), os.path.join(subdir, fname))
    self._check_file(find_file(fname, ext='.py'), os.path.join(self.tmpdir, fname))
    self._check_file(find_file(fname, ext=['.py']), os.path.join(self.tmpdir, fname))
    self._check_file(find_file(fname[:-3], ext='.py'), os.path.join(self.tmpdir, fname))
    self._check_file(find_file(fname[:-3], ext=['.py']), os.path.join(self.tmpdir, fname))
    self._check_file(find_file(subdir_name, pathlist=[self.tmpdir, subdir], cwd=False), os.path.join(subdir, subdir_name))
    self._check_file(find_file(subdir_name, pathlist=['', self.tmpdir, subdir], cwd=False), os.path.join(subdir, subdir_name))