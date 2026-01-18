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
def test_create_rename(self):
    """Rename an inventory entry while creating the file"""
    tree = self.make_branch_and_tree('.')
    with open('name1', 'wb') as f:
        f.write(b'Hello')
    tree.add('name1')
    tree.commit(message='hello')
    tree.rename_one('name1', 'name2')
    os.unlink('name2')
    transform_tree(tree, tree.branch.basis_tree())