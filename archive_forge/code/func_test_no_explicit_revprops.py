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
def test_no_explicit_revprops(self):
    branch, tt = self.get_branch_and_transform()
    rev_id = tt.commit(branch, 'message', authors=['Author1 <author1@example.com>', 'Author2 <author2@example.com>'])
    revision = branch.repository.get_revision(rev_id)
    self.assertEqual(['Author1 <author1@example.com>', 'Author2 <author2@example.com>'], revision.get_apparent_authors())
    self.assertEqual('tree', revision.properties['branch-nick'])