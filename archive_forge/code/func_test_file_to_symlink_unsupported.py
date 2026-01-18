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
def test_file_to_symlink_unsupported(self):
    wt = self.make_branch_and_tree('.')
    self.build_tree(['foo'])
    wt.add(['foo'])
    wt.commit('one')
    self.overrideAttr(osutils, 'supports_symlinks', lambda p: False)
    tt = wt.transform()
    self.addCleanup(tt.finalize)
    foo_trans_id = tt.trans_id_tree_path('foo')
    tt.delete_contents(foo_trans_id)
    log = BytesIO()
    trace.push_log_file(log)
    tt.create_symlink('bar', foo_trans_id)
    tt.apply()
    self.assertContainsRe(log.getvalue(), b'Unable to create symlink "foo" on this filesystem')