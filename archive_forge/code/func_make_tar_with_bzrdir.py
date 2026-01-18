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
def make_tar_with_bzrdir(self):
    result = BytesIO()
    with tarfile.open('tar-with-bzrdir.tar', 'w', result) as tar_file:
        os.mkdir('toplevel-dir')
        tar_file.add('toplevel-dir')
        os.mkdir('toplevel-dir/.bzr')
        tar_file.add('toplevel-dir/.bzr')
    rmtree('toplevel-dir')
    result.seek(0)
    return result