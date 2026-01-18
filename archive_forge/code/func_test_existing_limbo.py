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
def test_existing_limbo(self):
    transform, root = self.transform()
    limbo_name = transform._limbodir
    deletion_path = transform._deletiondir
    os.mkdir(pathjoin(limbo_name, 'hehe'))
    self.assertRaises(ImmortalLimbo, transform.apply)
    self.assertRaises(LockError, self.wt.unlock)
    self.assertRaises(ExistingLimbo, self.transform)
    self.assertRaises(LockError, self.wt.unlock)
    os.rmdir(pathjoin(limbo_name, 'hehe'))
    os.rmdir(limbo_name)
    os.rmdir(deletion_path)
    transform, root = self.transform()
    transform.apply()