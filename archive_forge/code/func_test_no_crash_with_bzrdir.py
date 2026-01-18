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
def test_no_crash_with_bzrdir(self):
    tar_file = self.make_tar_with_bzrdir()
    tree = ControlDir.create_standalone_workingtree('tree')
    import_tar(tree, tar_file)