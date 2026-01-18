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
def test_broken_tar(self):

    def builder(fileobj, mode='w'):
        return tarfile.open('project-0.1.tar', mode, fileobj)
    self.archive_test(builder, import_tar_broken, subdir=True)