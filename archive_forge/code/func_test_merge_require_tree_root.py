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
def test_merge_require_tree_root(self):
    tree = self.make_branch_and_tree('.')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    self.build_tree(['a'])
    tree.add('a')
    first_rev = tree.commit('added a')
    old_root_id = tree.path2id('')
    merger = _mod_merge.Merger.from_revision_ids(tree, _mod_revision.NULL_REVISION, first_rev)
    merger.merge_type = _mod_merge.Merge3Merger
    conflicts = merger.do_merge()
    self.assertEqual([], conflicts)
    self.assertEqual({''}, set(tree.all_versioned_paths()))
    tree.set_parent_ids([])