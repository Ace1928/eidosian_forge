from breezy import errors, gpg
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventory, versionedfile, vf_repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_corrupt_revision_access_asserts_if_reported_wrong(self):
    repo_url = self.get_url('inventory_with_unnecessary_ghost')
    repo = _mod_repository.Repository.open(repo_url)
    m = MatchesAncestry(repo, b'ghost')
    reported_wrong = False
    try:
        if m.match([b'the_ghost', b'ghost']) is not None:
            reported_wrong = True
    except errors.CorruptRepository:
        return
    if not reported_wrong:
        return
    self.assertRaises(errors.CorruptRepository, repo.get_revision, b'ghost')