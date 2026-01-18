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
def test_lca_supersedes_other_lca(self):
    """If one lca == base, the other lca takes precedence"""
    self.assertLCAMultiWay('this', 'bval', ['bval', 'lcaval'], 'lcaval', 'tval')
    self.assertLCAMultiWay('this', 'bval', ['bval', 'lcaval'], 'lcaval', 'bval')
    self.assertLCAMultiWay('other', 'bval', ['bval', 'lcaval'], 'bval', 'lcaval')
    self.assertLCAMultiWay('conflict', 'bval', ['bval', 'lcaval'], 'bval', 'tval')