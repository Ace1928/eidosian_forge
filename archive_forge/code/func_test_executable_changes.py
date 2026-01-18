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
def test_executable_changes(self):
    builder = self.get_builder()
    builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('foo', b'foo-id', 'file', b'a\nb\nc\n'))], revision_id=b'A-id')
    builder.build_snapshot([b'A-id'], [], revision_id=b'C-id')
    builder.build_snapshot([b'A-id'], [], revision_id=b'B-id')
    builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
    builder.build_snapshot([b'C-id', b'B-id'], [], revision_id=b'E-id')
    wt = self.get_wt_from_builder(builder)
    with wt.transform() as tt:
        tt.set_executability(True, tt.trans_id_tree_path('foo'))
        tt.apply()
    self.assertTrue(wt.is_executable('foo'))
    wt.commit('F-id', rev_id=b'F-id')
    wt.set_parent_ids([b'D-id'])
    wt.branch.set_last_revision_info(3, b'D-id')
    wt.revert()
    self.assertFalse(wt.is_executable('foo'))
    conflicts = wt.merge_from_branch(wt.branch, to_revision=b'F-id')
    self.assertEqual(0, len(conflicts))
    self.assertTrue(wt.is_executable('foo'))