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
def test_merge_uncommitted_otherbasis_ancestor_of_thisbasis_weave(self):
    tree_a = self.make_branch_and_tree('a')
    self.build_tree(['a/file_1', 'a/file_2'])
    tree_a.add(['file_1'])
    tree_a.commit('commit 1')
    tree_a.add(['file_2'])
    tree_a.commit('commit 2')
    tree_b = tree_a.controldir.sprout('b').open_workingtree()
    tree_b.rename_one('file_1', 'renamed')
    merger = _mod_merge.Merger.from_uncommitted(tree_a, tree_b)
    merger.merge_type = _mod_merge.WeaveMerger
    merger.do_merge()
    self.assertEqual(tree_a.get_parent_ids(), [tree_b.last_revision()])