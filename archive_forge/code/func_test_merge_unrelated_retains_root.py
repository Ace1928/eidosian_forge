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
def test_merge_unrelated_retains_root(self):
    wt = self.make_branch_and_tree('tree')
    other_tree = self.make_branch_and_tree('other')
    self.addCleanup(other_tree.lock_read().unlock)
    merger = _mod_merge.Merge3Merger(wt, wt, wt.basis_tree(), other_tree, this_branch=wt.branch, do_merge=False)
    with wt.preview_transform() as merger.tt:
        merger._compute_transform()
        new_root_id = merger.tt.final_file_id(merger.tt.root)
        self.assertEqual(wt.path2id(''), new_root_id)