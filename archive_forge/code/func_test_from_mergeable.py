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
def test_from_mergeable(self):
    this, other = self.prepare_for_merging()
    md = merge_directive.MergeDirective2.from_objects(other.branch.repository, b'rev3', 0, 0, 'this')
    other.lock_read()
    self.addCleanup(other.unlock)
    merger, verified = Merger.from_mergeable(this, md)
    md.patch = None
    merger, verified = Merger.from_mergeable(this, md)
    self.assertEqual('inapplicable', verified)
    self.assertEqual(b'rev3', merger.other_rev_id)
    self.assertEqual(b'rev1', merger.base_rev_id)
    md.base_revision_id = b'rev2b'
    merger, verified = Merger.from_mergeable(this, md)
    self.assertEqual(b'rev2b', merger.base_rev_id)