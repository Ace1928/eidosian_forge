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
def test_file_not_in_one_lca(self):
    builder = self.get_builder()
    builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None))], revision_id=b'A-id')
    builder.build_snapshot([b'A-id'], [], revision_id=b'B-id')
    builder.build_snapshot([b'A-id'], [('add', ('a', b'a-id', 'file', b'a\nb\nc\n'))], revision_id=b'C-id')
    builder.build_snapshot([b'C-id', b'B-id'], [], revision_id=b'E-id')
    builder.build_snapshot([b'B-id', b'C-id'], [('add', ('a', b'a-id', 'file', b'a\nb\nc\n'))], revision_id=b'D-id')
    merge_obj = self.make_merge_obj(builder, b'E-id')
    self.assertEqual([b'B-id', b'C-id'], [t.get_revision_id() for t in merge_obj._lca_trees])
    self.assertEqual(b'A-id', merge_obj.base_tree.get_revision_id())
    entries = list(merge_obj._entries_lca())
    self.assertEqual([], entries)