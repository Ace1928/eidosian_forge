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
def prepare_cherrypick(self):
    """Prepare a pair of trees for cherrypicking tests.

        Both trees have a file, 'file'.
        rev1 sets content to 'a'.
        rev2b adds 'b'.
        rev3b adds 'c'.
        A full merge of rev2b and rev3b into this_tree would add both 'b' and
        'c'.  A successful cherrypick of rev2b-rev3b into this_tree will add
        'c', but not 'b'.
        """
    this_tree = self.make_branch_and_tree('this')
    self.build_tree_contents([('this/file', b'a\n')])
    this_tree.add('file')
    this_tree.commit('rev1')
    other_tree = this_tree.controldir.sprout('other').open_workingtree()
    self.build_tree_contents([('other/file', b'a\nb\n')])
    other_tree.commit('rev2b', rev_id=b'rev2b')
    self.build_tree_contents([('other/file', b'c\na\nb\n')])
    other_tree.commit('rev3b', rev_id=b'rev3b')
    this_tree.lock_write()
    self.addCleanup(this_tree.unlock)
    return (this_tree, other_tree)