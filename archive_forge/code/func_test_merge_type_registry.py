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
def test_merge_type_registry(self):
    merge_type_option = option.Option.OPTIONS['merge-type']
    self.assertFalse('merge4' in [x[0] for x in merge_type_option.iter_switches()])
    registry = _mod_merge.get_merge_type_registry()
    registry.register_lazy('merge4', 'breezy.merge', 'Merge4Merger', 'time-travelling merge')
    self.assertTrue('merge4' in [x[0] for x in merge_type_option.iter_switches()])
    registry.remove('merge4')
    self.assertFalse('merge4' in [x[0] for x in merge_type_option.iter_switches()])