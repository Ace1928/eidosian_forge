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
def test_new_file_caches_sha1(self):
    trans, root, contents, sha1 = self.transform_for_sha1_test()
    trans_id = trans.new_file('file1', root, contents, file_id=b'file1-id', sha1=sha1)
    st_val = osutils.lstat(trans._limbo_name(trans_id))
    o_sha1, o_st_val = trans._observed_sha1s[trans_id]
    self.assertEqual(o_sha1, sha1)
    self.assertEqualStat(o_st_val, st_val)