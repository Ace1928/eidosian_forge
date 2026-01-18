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
def assertTreeEntriesEqual(self, expected_entries, tree):
    """Assert that 'tree' contains the expected inventory entries.

        :param expected_entries: sequence of (path, file-id) pairs.
        """
    files = [(path, ie.file_id) for path, ie in tree.iter_entries_by_dir()]
    self.assertEqual(expected_entries, files)