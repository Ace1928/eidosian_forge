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
@unittest.skipUnless(ZIP_SUPPORT, 'Need zip support to run')
def test_make_zipfile_no_zlib(self):
    patch(self, archive_util.zipfile, 'zlib', None)
    called = []
    zipfile_class = zipfile.ZipFile

    def fake_zipfile(*a, **kw):
        if kw.get('compression', None) == zipfile.ZIP_STORED:
            called.append((a, kw))
        return zipfile_class(*a, **kw)
    patch(self, archive_util.zipfile, 'ZipFile', fake_zipfile)
    tmpdir = self._create_files()
    base_name = os.path.join(self.mkdtemp(), 'archive')
    with change_cwd(tmpdir):
        make_zipfile(base_name, 'dist')
    tarball = base_name + '.zip'
    self.assertEqual(called, [((tarball, 'w'), {'compression': zipfile.ZIP_STORED})])
    self.assertTrue(os.path.exists(tarball))
    with zipfile.ZipFile(tarball) as zf:
        self.assertEqual(sorted(zf.namelist()), self._zip_created_files)