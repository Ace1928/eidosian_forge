import sys
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy import tests, transform
from breezy.bzr import inventory, remote
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_fileids_altered_between_two_revs(self):
    self.branch.lock_read()
    self.addCleanup(self.branch.unlock)
    self.branch.repository.fileids_altered_by_revision_ids([b'rev-J', b'rev-K'])
    self.assertEqual({b'b-file-id-2006-01-01-defg': {b'rev-J'}, b'c-funky<file-id>quiji%bo': {b'rev-K'}}, self.branch.repository.fileids_altered_by_revision_ids([b'rev-J', b'rev-K']))
    self.assertEqual({b'b-file-id-2006-01-01-defg': {b'rev-<D>'}, b'file-d': {b'rev-F'}}, self.branch.repository.fileids_altered_by_revision_ids([b'rev-<D>', b'rev-F']))
    self.assertEqual({b'b-file-id-2006-01-01-defg': {b'rev-<D>', b'rev-G', b'rev-J'}, b'c-funky<file-id>quiji%bo': {b'rev-K'}, b'file-d': {b'rev-F'}}, self.branch.repository.fileids_altered_by_revision_ids([b'rev-<D>', b'rev-G', b'rev-F', b'rev-K', b'rev-J']))
    self.assertEqual({b'a-file-id-2006-01-01-abcd': {b'rev-B'}, b'b-file-id-2006-01-01-defg': {b'rev-<D>', b'rev-G', b'rev-J'}, b'c-funky<file-id>quiji%bo': {b'rev-K'}, b'file-d': {b'rev-F'}}, self.branch.repository.fileids_altered_by_revision_ids([b'rev-G', b'rev-F', b'rev-C', b'rev-B', b'rev-<D>', b'rev-K', b'rev-J']))