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
def test_find_base_new_root_criss_cross(self):
    builder = self.get_builder()
    builder.build_snapshot(None, [('add', ('', None, 'directory', None))], revision_id=b'A-id')
    builder.build_snapshot([], [('add', ('', None, 'directory', None))], revision_id=b'B-id')
    builder.build_snapshot([b'A-id', b'B-id'], [], revision_id=b'D-id')
    builder.build_snapshot([b'A-id', b'B-id'], [], revision_id=b'C-id')
    merger = self.make_Merger(builder, b'D-id')
    self.assertEqual(b'A-id', merger.base_rev_id)
    self.assertTrue(merger._is_criss_cross)
    self.assertEqual([b'A-id', b'B-id'], [t.get_revision_id() for t in merger._lca_trees])