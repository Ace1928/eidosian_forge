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
def test_criss_cross_not_supported_merge_type(self):
    merger = self.make_Merger(self.setup_criss_cross_graph(), b'E-id')
    merger.merge_type = LoggingMerger
    merge_obj = merger.make_merger()
    self.assertIsInstance(merge_obj, LoggingMerger)
    self.assertFalse('lca_trees' in merge_obj.kwargs)