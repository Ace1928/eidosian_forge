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
def test_check_metadata_deprecated(self):
    dist, cmd = self.get_cmd()
    with check_warnings() as w:
        warnings.simplefilter('always')
        cmd.check_metadata()
        self.assertEqual(len(w.warnings), 1)