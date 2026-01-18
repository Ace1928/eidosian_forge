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
def test_create_files_same_timestamp(self):
    transform, root = self.transform()
    self.wt.lock_tree_write()
    self.addCleanup(self.wt.unlock)
    transform._creation_mtime = creation_mtime = time.time() - 20.0
    transform.create_file([b'content-one'], transform.create_path('one', root))
    time.sleep(1)
    transform.create_file([b'content-two'], transform.create_path('two', root))
    transform.apply()
    fo, st1 = self.wt.get_file_with_stat('one', filtered=False)
    fo.close()
    fo, st2 = self.wt.get_file_with_stat('two', filtered=False)
    fo.close()
    self.assertTrue(abs(creation_mtime - st1.st_mtime) < 2.0, '{} != {} within 2 seconds'.format(creation_mtime, st1.st_mtime))
    self.assertEqual(st1.st_mtime, st2.st_mtime)