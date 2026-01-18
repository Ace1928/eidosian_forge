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
@unittest.skipUnless(ZLIB_SUPPORT, 'requires zlib')
@unittest.skipUnless(UID_GID_SUPPORT, 'Requires grp and pwd support')
@unittest.skipIf(find_executable('tar') is None, 'The tar command is not found')
@unittest.skipIf(find_executable('gzip') is None, 'The gzip command is not found')
def test_make_distribution_owner_group(self):
    dist, cmd = self.get_cmd()
    cmd.formats = ['gztar']
    cmd.owner = pwd.getpwuid(0)[0]
    cmd.group = grp.getgrgid(0)[0]
    cmd.ensure_finalized()
    cmd.run()
    archive_name = join(self.tmp_dir, 'dist', 'fake-1.0.tar.gz')
    archive = tarfile.open(archive_name)
    try:
        for member in archive.getmembers():
            self.assertEqual(member.uid, 0)
            self.assertEqual(member.gid, 0)
    finally:
        archive.close()
    dist, cmd = self.get_cmd()
    cmd.formats = ['gztar']
    cmd.ensure_finalized()
    cmd.run()
    archive_name = join(self.tmp_dir, 'dist', 'fake-1.0.tar.gz')
    archive = tarfile.open(archive_name)
    try:
        for member in archive.getmembers():
            self.assertEqual(member.uid, os.getuid())
    finally:
        archive.close()