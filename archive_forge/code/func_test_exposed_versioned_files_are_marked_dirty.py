from breezy import errors, gpg
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventory, versionedfile, vf_repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_exposed_versioned_files_are_marked_dirty(self):
    repo = self.make_repository('.')
    repo.lock_write()
    signatures = repo.signatures
    revisions = repo.revisions
    inventories = repo.inventories
    repo.unlock()
    self.assertRaises(errors.ObjectNotLocked, signatures.keys)
    self.assertRaises(errors.ObjectNotLocked, revisions.keys)
    self.assertRaises(errors.ObjectNotLocked, inventories.keys)
    self.assertRaises(errors.ObjectNotLocked, signatures.add_lines, ('foo',), [], [])
    self.assertRaises(errors.ObjectNotLocked, revisions.add_lines, ('foo',), [], [])
    self.assertRaises(errors.ObjectNotLocked, inventories.add_lines, ('foo',), [], [])