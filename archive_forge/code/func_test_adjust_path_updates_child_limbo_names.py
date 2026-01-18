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
def test_adjust_path_updates_child_limbo_names(self):
    tree = self.make_branch_and_tree('tree')
    transform = tree.transform()
    self.addCleanup(transform.finalize)
    foo_id = transform.new_directory('foo', transform.root)
    bar_id = transform.new_directory('bar', foo_id)
    baz_id = transform.new_directory('baz', bar_id)
    qux_id = transform.new_directory('qux', baz_id)
    transform.adjust_path('quxx', foo_id, bar_id)
    self.assertStartsWith(transform._limbo_name(qux_id), transform._limbo_name(bar_id))