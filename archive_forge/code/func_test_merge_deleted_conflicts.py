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
def test_merge_deleted_conflicts(self):
    wta = self.make_branch_and_tree('a')
    with open('a/file', 'wb') as f:
        f.write(b'contents\n')
    wta.add('file')
    wta.commit('a_revision', allow_pointless=False)
    self.run_bzr('branch a b')
    os.remove('a/file')
    wta.commit('removed file', allow_pointless=False)
    with open('b/file', 'wb') as f:
        f.write(b'changed contents\n')
    wtb = WorkingTree.open('b')
    wtb.commit('changed file', allow_pointless=False)
    wtb.merge_from_branch(wta.branch, wta.branch.last_revision(), wta.branch.get_rev_id(1))
    self.assertFalse(os.path.lexists('b/file'))