import unittest
import os
import sys
import tarfile
from os.path import splitdrive
import warnings
from distutils import archive_util
from distutils.archive_util import (check_archive_formats, make_tarball,
from distutils.spawn import find_executable, spawn
from distutils.tests import support
from test.support import patch
from test.support.os_helper import change_cwd
from test.support.warnings_helper import check_warnings
@unittest.skipUnless(ZLIB_SUPPORT, 'Need zlib support to run')
def test_make_tarball(self, name='archive'):
    tmpdir = self._create_files()
    self._make_tarball(tmpdir, name, '.tar.gz')
    self._make_tarball(tmpdir, name, '.tar', compress=None)