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
def assertRemoveExternalReferences(self, filtered_parent_map, child_map, tails, parent_map):
    """Assert results for _PlanMerge._remove_external_references."""
    act_filtered_parent_map, act_child_map, act_tails = _PlanMerge._remove_external_references(parent_map)
    self.assertEqual(filtered_parent_map, act_filtered_parent_map)
    self.assertEqual(child_map, act_child_map)
    self.assertEqual(sorted(tails), sorted(act_tails))