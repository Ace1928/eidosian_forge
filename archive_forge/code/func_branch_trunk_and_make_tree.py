import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def branch_trunk_and_make_tree(self, trunk_repo, relpath):
    tree = self.make_branch_and_memory_tree('branch')
    trunk_repo.lock_read()
    self.addCleanup(trunk_repo.unlock)
    tree.branch.repository.fetch(trunk_repo, revision_id=b'rev-1')
    tree.set_parent_ids([b'rev-1'])
    return tree