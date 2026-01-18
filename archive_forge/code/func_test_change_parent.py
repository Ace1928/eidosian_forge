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
def test_change_parent(self):
    """Ensure that after we change a parent, the results are still right.

        Renames and parent changes on pending transforms can happen as part
        of conflict resolution, and are explicitly permitted by the
        TreeTransform API.

        This test ensures they work correctly with the rename-avoidance
        optimization.
        """
    transform, root = self.transform()
    parent1 = transform.new_directory('parent1', root)
    child1 = transform.new_file('child1', parent1, [b'contents'])
    parent2 = transform.new_directory('parent2', root)
    transform.adjust_path('child1', parent2, child1)
    transform.apply()
    self.assertPathDoesNotExist(self.wt.abspath('parent1/child1'))
    self.assertPathExists(self.wt.abspath('parent2/child1'))
    self.assertEqual(2, transform.rename_count)