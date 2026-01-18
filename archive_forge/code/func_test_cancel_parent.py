import errno
import os
import sys
import time
from io import BytesIO
from breezy.bzr.transform import resolve_checkout
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from ... import osutils, tests, trace, transform, urlutils
from ...bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ...errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ...osutils import file_kind, pathjoin
from ...transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ...transport import FileExists
from ...tree import TreeChange
from .. import TestSkipped, features
from ..features import HardlinkFeature, SymlinkFeature
def test_cancel_parent(self):
    """Cancelling a parent doesn't cause deletion of a non-empty directory

        This is like the test_change_parent, except that we cancel the parent
        before adjusting the path.  The transform must detect that the
        directory is non-empty, and move children to safe locations.
        """
    transform, root = self.transform()
    parent1 = transform.new_directory('parent1', root)
    child1 = transform.new_file('child1', parent1, [b'contents'])
    child2 = transform.new_file('child2', parent1, [b'contents'])
    try:
        transform.cancel_creation(parent1)
    except OSError:
        self.fail('Failed to move child1 before deleting parent1')
    transform.cancel_creation(child2)
    transform.create_directory(parent1)
    try:
        transform.cancel_creation(parent1)
    except OSError:
        self.fail('Transform still thinks child2 is a child of parent1')
    parent2 = transform.new_directory('parent2', root)
    transform.adjust_path('child1', parent2, child1)
    transform.apply()
    self.assertPathDoesNotExist(self.wt.abspath('parent1'))
    self.assertPathExists(self.wt.abspath('parent2/child1'))
    self.assertEqual(2, transform.rename_count)