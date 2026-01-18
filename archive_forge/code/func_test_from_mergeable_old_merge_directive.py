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
def test_from_mergeable_old_merge_directive(self):
    this, other = self.prepare_for_merging()
    other.lock_write()
    self.addCleanup(other.unlock)
    md = merge_directive.MergeDirective.from_objects(other.branch.repository, b'rev3', 0, 0, 'this')
    merger, verified = Merger.from_mergeable(this, md)
    self.assertEqual(b'rev3', merger.other_rev_id)
    self.assertEqual(b'rev1', merger.base_rev_id)