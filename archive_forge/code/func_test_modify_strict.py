import codecs
import errno
import os
import sys
import time
from io import BytesIO, StringIO
import fastbencode as bencode
from .. import filters, osutils
from .. import revision as _mod_revision
from .. import rules, tests, trace, transform, urlutils
from ..bzr import generate_ids
from ..bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ..controldir import ControlDir
from ..diff import show_diff_trees
from ..errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ..merge import Merge3Merger, Merger
from ..mutabletree import MutableTree
from ..osutils import file_kind, pathjoin
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..transport import FileExists
from . import TestCaseInTempDir, TestSkipped, features
from .features import HardlinkFeature, SymlinkFeature
def test_modify_strict(self):
    branch, tt = self.get_branch_and_transform()
    tt.new_file('file', tt.root, [b'contents'], b'file-id')
    tt.commit(branch, 'message', strict=True)
    tt = branch.basis_tree().preview_transform()
    self.addCleanup(tt.finalize)
    trans_id = tt.trans_id_file_id(b'file-id')
    tt.delete_contents(trans_id)
    tt.create_file([b'contents'], trans_id)
    tt.commit(branch, 'message', strict=True)