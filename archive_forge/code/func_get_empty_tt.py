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
def get_empty_tt(self):
    b = self.make_repository('foo')
    null_tree = b.revision_tree(_mod_revision.NULL_REVISION)
    tt = null_tree.preview_transform()
    tt.new_directory('', transform.ROOT_PARENT, b'tree-root')
    tt.fixup_new_roots()
    self.addCleanup(tt.finalize)
    return tt