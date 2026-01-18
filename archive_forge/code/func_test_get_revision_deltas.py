from breezy import errors, gpg
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventory, versionedfile, vf_repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_get_revision_deltas(self):
    repository = self.controldir.open_repository()
    repository.lock_read()
    self.addCleanup(repository.unlock)
    revisions = [repository.get_revision(r) for r in [b'rev1', b'rev2', b'rev3', b'rev4']]
    deltas1 = list(repository.get_revision_deltas(revisions))
    deltas2 = [repository.get_revision_delta(r.revision_id) for r in revisions]
    self.assertEqual(deltas1, deltas2)