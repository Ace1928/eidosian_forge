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
def test_apply_deletions(self):
    self.build_tree(['a/', 'b/'])
    mover = _FileMover()
    mover.pre_delete('a', 'q')
    mover.pre_delete('b', 'r')
    self.assertPathExists('q')
    self.assertPathExists('r')
    self.assertPathDoesNotExist('a')
    self.assertPathDoesNotExist('b')
    mover.apply_deletions()
    self.assertPathDoesNotExist('q')
    self.assertPathDoesNotExist('r')
    self.assertPathDoesNotExist('a')
    self.assertPathDoesNotExist('b')