from breezy import errors, gpg
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventory, versionedfile, vf_repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_install_revisions(self):
    wt = self.make_branch_and_tree('source')
    wt.commit('A', allow_pointless=True, rev_id=b'A')
    repo = wt.branch.repository
    repo.lock_write()
    repo.start_write_group()
    repo.sign_revision(b'A', gpg.LoopbackGPGStrategy(None))
    repo.commit_write_group()
    repo.unlock()
    repo.lock_read()
    self.addCleanup(repo.unlock)
    repo2 = self.make_repository('repo2')
    revision = repo.get_revision(b'A')
    tree = repo.revision_tree(b'A')
    signature = repo.get_signature_text(b'A')
    repo2.lock_write()
    self.addCleanup(repo2.unlock)
    vf_repository.install_revisions(repo2, [(revision, tree, signature)])
    self.assertEqual(revision, repo2.get_revision(b'A'))
    self.assertEqual(signature, repo2.get_signature_text(b'A'))