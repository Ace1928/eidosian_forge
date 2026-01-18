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
def setup_merge_with_base(self):
    self.add_rev(b'root', b'COMMON', [], b'abc')
    self.add_rev(b'root', b'THIS', [b'COMMON'], b'abcd')
    self.add_rev(b'root', b'BASE', [b'COMMON'], b'eabc')
    self.add_rev(b'root', b'OTHER', [b'BASE'], b'eafb')