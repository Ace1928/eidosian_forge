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
def test_first_commit_with_merge_parents(self):
    branch = self.make_branch('branch')
    branch.lock_write()
    self.addCleanup(branch.unlock)
    tt = branch.basis_tree().preview_transform()
    self.addCleanup(tt.finalize)
    e = self.assertRaises(ValueError, tt.commit, branch, 'my message', [b'rev1b-id'])
    self.assertEqual('Cannot supply merge parents for first commit.', str(e))
    self.assertEqual(_mod_revision.NULL_REVISION, branch.last_revision())