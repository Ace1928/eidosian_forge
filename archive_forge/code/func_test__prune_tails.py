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
def test__prune_tails(self):
    self.assertPruneTails({1: [], 2: [], 3: []}, [], {1: [], 2: [], 3: []})
    self.assertPruneTails({1: [], 3: []}, [2], {1: [], 2: [], 3: []})
    self.assertPruneTails({1: []}, [3], {1: [], 2: [3], 3: []})
    self.assertPruneTails({1: []}, [5], {1: [], 2: [3, 4], 3: [5], 4: [5], 5: []})
    self.assertPruneTails({1: [6], 6: []}, [5], {1: [2, 6], 2: [3, 4], 3: [5], 4: [5], 5: [], 6: []})
    self.assertPruneTails({1: [3], 3: []}, [4, 5], {1: [2, 3], 2: [4, 5], 3: [], 4: [], 5: []})
    self.assertPruneTails({1: [3], 3: []}, [5, 4], {1: [2, 3], 2: [4, 5], 3: [], 4: [], 5: []})