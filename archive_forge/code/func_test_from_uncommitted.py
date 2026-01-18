import os
import sys
import tempfile
import breezy
from .. import controldir, errors, merge_directive, osutils
from ..bzr import generate_ids
from ..bzr.conflicts import ContentsConflict, PathConflict, TextConflict
from ..merge import Diff3Merger, Merge3Merger, Merger, WeaveMerger
from ..osutils import getcwd, pathjoin
from ..workingtree import WorkingTree
from . import TestCaseWithTransport, TestSkipped, features
def test_from_uncommitted(self):
    this, other = self.set_up_trees()
    merger = Merger.from_uncommitted(this, other, None)
    self.assertIs(other, merger.other_tree)
    self.assertIs(None, merger.other_rev_id)
    self.assertEqual(b'rev2b', merger.base_rev_id)