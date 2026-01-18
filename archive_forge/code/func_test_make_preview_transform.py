import contextlib
import os
from .. import branch as _mod_branch
from .. import conflicts, errors, memorytree
from .. import merge as _mod_merge
from .. import option
from .. import revision as _mod_revision
from .. import tests, transform
from ..bzr import inventory, knit, versionedfile
from ..bzr.conflicts import (ContentsConflict, DeletingParent, MissingParent,
from ..conflicts import ConflictList
from ..errors import NoCommits, UnrelatedBranches
from ..merge import _PlanMerge, merge_inner, transform_tree
from ..osutils import basename, file_kind, pathjoin
from ..workingtree import PointlessMerge, WorkingTree
from . import (TestCaseWithMemoryTransport, TestCaseWithTransport, features,
def test_make_preview_transform(self):
    this_tree = self.make_branch_and_tree('this')
    self.build_tree_contents([('this/file', b'1\n')])
    this_tree.add('file', ids=b'file-id')
    this_tree.commit('rev1', rev_id=b'rev1')
    other_tree = this_tree.controldir.sprout('other').open_workingtree()
    self.build_tree_contents([('this/file', b'1\n2a\n')])
    this_tree.commit('rev2', rev_id=b'rev2a')
    self.build_tree_contents([('other/file', b'2b\n1\n')])
    other_tree.commit('rev2', rev_id=b'rev2b')
    this_tree.lock_write()
    self.addCleanup(this_tree.unlock)
    merger = _mod_merge.Merger.from_revision_ids(this_tree, b'rev2b', other_branch=other_tree.branch)
    merger.merge_type = _mod_merge.Merge3Merger
    tree_merger = merger.make_merger()
    with tree_merger.make_preview_transform() as tt:
        preview_tree = tt.get_preview_tree()
        with this_tree.get_file('file') as tree_file:
            self.assertEqual(b'1\n2a\n', tree_file.read())
        with preview_tree.get_file('file') as preview_file:
            self.assertEqual(b'2b\n1\n2a\n', preview_file.read())