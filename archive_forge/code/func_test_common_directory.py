import os
import tarfile
import tempfile
import warnings
from io import BytesIO
from shutil import copy2, copytree, rmtree
from .. import osutils
from .. import revision as _mod_revision
from .. import transform
from ..controldir import ControlDir
from ..export import export
from ..upstream_import import (NotArchiveType, ZipFileWrapper,
from . import TestCaseInTempDir, TestCaseWithTransport
from .features import UnicodeFilenameFeature
def test_common_directory(self):
    self.assertEqual(common_directory(['ab/c/d', 'ab/c/e']), 'ab')
    self.assertIs(common_directory(['ab/c/d', 'ac/c/e']), None)
    self.assertEqual('FEEDME', common_directory(['FEEDME']))