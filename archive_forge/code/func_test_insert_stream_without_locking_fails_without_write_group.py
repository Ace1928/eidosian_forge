import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_insert_stream_without_locking_fails_without_write_group(self):
    repo = self.make_repository('test-repo')
    self.addCleanup(repo.lock_write().unlock)
    sink = repo._get_sink()
    stream = [('texts', [versionedfile.FulltextContentFactory((b'file-id', b'rev-id'), (), None, b'lines\n')])]
    self.assertRaises(errors.BzrError, sink.insert_stream_without_locking, stream, repo._format)