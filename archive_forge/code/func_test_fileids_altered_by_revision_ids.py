import sys
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy import tests, transform
from breezy.bzr import inventory, remote
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_fileids_altered_by_revision_ids(self):
    self.branch.lock_read()
    self.addCleanup(self.branch.unlock)
    self.assertEqual({b'a-file-id-2006-01-01-abcd': {b'rev-A'}, b'b-file-id-2006-01-01-defg': {b'rev-A'}, b'c-funky<file-id>quiji%bo': {b'rev-A'}}, self.fileids_altered_by_revision_ids([b'rev-A']))
    self.assertEqual({b'a-file-id-2006-01-01-abcd': {b'rev-B'}}, self.branch.repository.fileids_altered_by_revision_ids([b'rev-B']))
    self.assertEqual({b'b-file-id-2006-01-01-defg': {b'rev-<D>'}}, self.branch.repository.fileids_altered_by_revision_ids([b'rev-<D>']))