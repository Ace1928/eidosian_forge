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
def test__remove_external_references(self):
    self.assertRemoveExternalReferences({3: [2], 2: [1], 1: []}, {1: [2], 2: [3], 3: []}, [1], {3: [2], 2: [1], 1: []})
    self.assertRemoveExternalReferences({1: [2], 2: [3], 3: []}, {3: [2], 2: [1], 1: []}, [3], {1: [2], 2: [3], 3: []})
    self.assertRemoveExternalReferences({3: [2], 2: [1], 1: []}, {1: [2], 2: [3], 3: []}, [1], {3: [2, 4], 2: [1, 5], 1: [6]})
    self.assertRemoveExternalReferences({4: [2, 3], 3: [], 2: [1], 1: []}, {1: [2], 2: [4], 3: [4], 4: []}, [1, 3], {4: [2, 3], 3: [5], 2: [1], 1: [6]})
    self.assertRemoveExternalReferences({1: [3], 2: [3, 4], 3: [], 4: []}, {1: [], 2: [], 3: [1, 2], 4: [2]}, [3, 4], {1: [3], 2: [3, 4], 3: [5], 4: []})