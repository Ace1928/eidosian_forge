import os
import tarfile
import unittest
import warnings
import zipfile
from os.path import join
from textwrap import dedent
from test.support import captured_stdout
from test.support.warnings_helper import check_warnings
from distutils.command.sdist import sdist, show_formats
from distutils.core import Distribution
from distutils.tests.test_config import BasePyPIRCCommandTestCase
from distutils.errors import DistutilsOptionError
from distutils.spawn import find_executable
from distutils.log import WARN
from distutils.filelist import FileList
from distutils.archive_util import ARCHIVE_FORMATS
from distutils.core import setup
import somecode
@unittest.skipUnless(ZLIB_SUPPORT, 'Need zlib support to run')
def test_manual_manifest(self):
    dist, cmd = self.get_cmd()
    cmd.formats = ['gztar']
    cmd.ensure_finalized()
    self.write_file((self.tmp_dir, cmd.manifest), 'README.manual')
    self.write_file((self.tmp_dir, 'README.manual'), 'This project maintains its MANIFEST file itself.')
    cmd.run()
    self.assertEqual(cmd.filelist.files, ['README.manual'])
    f = open(cmd.manifest)
    try:
        manifest = [line.strip() for line in f.read().split('\n') if line.strip() != '']
    finally:
        f.close()
    self.assertEqual(manifest, ['README.manual'])
    archive_name = join(self.tmp_dir, 'dist', 'fake-1.0.tar.gz')
    archive = tarfile.open(archive_name)
    try:
        filenames = [tarinfo.name for tarinfo in archive]
    finally:
        archive.close()
    self.assertEqual(sorted(filenames), ['fake-1.0', 'fake-1.0/PKG-INFO', 'fake-1.0/README.manual'])