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
def test_commit_malformed(self):
    """Committing a malformed transform should raise an exception.

        In this case, we are adding a file without adding its parent.
        """
    branch, tt = self.get_branch_and_transform()
    parent_id = tt.trans_id_file_id(b'parent-id')
    tt.new_file('file', parent_id, [b'contents'], b'file-id')
    self.assertRaises(MalformedTransform, tt.commit, branch, 'message')