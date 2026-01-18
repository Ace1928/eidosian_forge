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
def test_find_base_triple_criss_cross(self):
    builder = self.setup_criss_cross_graph()
    builder.build_snapshot([b'A-id'], [], revision_id=b'F-id')
    builder.build_snapshot([b'E-id', b'F-id'], [], revision_id=b'H-id')
    builder.build_snapshot([b'D-id', b'F-id'], [], revision_id=b'G-id')
    merger = self.make_Merger(builder, b'H-id')
    self.assertEqual([b'B-id', b'C-id', b'F-id'], [t.get_revision_id() for t in merger._lca_trees])