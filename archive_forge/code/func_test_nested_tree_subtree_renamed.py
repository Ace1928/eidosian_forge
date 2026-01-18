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
def test_nested_tree_subtree_renamed(self):
    wt = self.make_branch_and_tree('tree', format='development-subtree')
    wt.lock_write()
    self.addCleanup(wt.unlock)
    sub_tree = self.make_branch_and_tree('tree/sub', format='development-subtree')
    wt.set_root_id(b'a-root-id')
    sub_tree.set_root_id(b'sub-tree-root')
    self.build_tree_contents([('tree/sub/file', b'text1')])
    sub_tree.add('file')
    sub_tree.commit('foo', rev_id=b'sub-A-id')
    wt.add_reference(sub_tree)
    wt.commit('set text to 1', rev_id=b'A-id', recursive=None)
    wt.commit('B', rev_id=b'B-id', recursive=None)
    wt.set_last_revision(b'A-id')
    wt.branch.set_last_revision_info(1, b'A-id')
    wt.commit('C', rev_id=b'C-id', recursive=None)
    wt.merge_from_branch(wt.branch, to_revision=b'B-id')
    wt.rename_one('sub', 'alt_sub')
    wt.commit('E', rev_id=b'E-id', recursive=None)
    wt.set_last_revision(b'B-id')
    wt.revert()
    wt.set_parent_ids([b'B-id', b'C-id'])
    wt.branch.set_last_revision_info(2, b'B-id')
    wt.commit('D', rev_id=b'D-id', recursive=None)
    merger = _mod_merge.Merger.from_revision_ids(wt, b'E-id')
    merger.merge_type = _mod_merge.Merge3Merger
    merge_obj = merger.make_merger()
    entries = list(merge_obj._entries_lca())
    root_id = b'a-root-id'
    self.assertEqual([(b'sub-tree-root', False, (('sub', ['sub', 'sub']), 'alt_sub', 'sub'), ((root_id, [root_id, root_id]), root_id, root_id), (('sub', ['sub', 'sub']), 'alt_sub', 'sub'), ((False, [False, False]), False, False), False)], entries)