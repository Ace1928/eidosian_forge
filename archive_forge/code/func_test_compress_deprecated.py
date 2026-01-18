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
@unittest.skipUnless(find_executable('compress'), 'The compress program is required')
def test_compress_deprecated(self):
    tmpdir = self._create_files()
    base_name = os.path.join(self.mkdtemp(), 'archive')
    old_dir = os.getcwd()
    os.chdir(tmpdir)
    try:
        with check_warnings() as w:
            warnings.simplefilter('always')
            make_tarball(base_name, 'dist', compress='compress')
    finally:
        os.chdir(old_dir)
    tarball = base_name + '.tar.Z'
    self.assertTrue(os.path.exists(tarball))
    self.assertEqual(len(w.warnings), 1)
    os.remove(tarball)
    old_dir = os.getcwd()
    os.chdir(tmpdir)
    try:
        with check_warnings() as w:
            warnings.simplefilter('always')
            make_tarball(base_name, 'dist', compress='compress', dry_run=True)
    finally:
        os.chdir(old_dir)
    self.assertFalse(os.path.exists(tarball))
    self.assertEqual(len(w.warnings), 1)