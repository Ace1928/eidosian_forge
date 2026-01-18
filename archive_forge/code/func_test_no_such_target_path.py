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
def test_no_such_target_path(self):
    """PathNotInTree is also raised if the specified path in the target
        tree does not exist.
        """
    dest_wt = self.setup_simple_branch('dest')
    self.setup_simple_branch('src', ['file.txt'])
    self.assertRaises(_mod_merge.PathNotInTree, self.do_merge_into, 'src', 'dest/no-such-dir/foo')
    dest_wt.lock_read()
    self.addCleanup(dest_wt.unlock)
    self.assertEqual([b'r1-dest'], dest_wt.get_parent_ids())
    self.assertTreeEntriesEqual([('', b'dest-root-id')], dest_wt)