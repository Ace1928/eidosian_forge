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
def test_change_root_id_add_files(self):
    if not self.workingtree_format.supports_setting_file_ids:
        raise tests.TestNotApplicable('format does not support setting file ids')
    transform, root = self.transform()
    self.assertNotEqual(b'new-root-id', self.wt.path2id(''))
    new_trans_id = transform.new_directory('', ROOT_PARENT, b'new-root-id')
    transform.new_file('file', new_trans_id, [b'new-contents\n'], b'new-file-id')
    transform.delete_contents(root)
    transform.unversion_file(root)
    transform.fixup_new_roots()
    transform.apply()
    self.assertEqual(b'new-root-id', self.wt.path2id(''))
    self.assertEqual(b'new-file-id', self.wt.path2id('file'))
    self.assertFileEqual(b'new-contents\n', self.wt.abspath('file'))