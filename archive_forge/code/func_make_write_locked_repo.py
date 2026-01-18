import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def make_write_locked_repo(self, relpath='repo'):
    repo = self.make_repository(relpath)
    repo.lock_write()
    self.addCleanup(repo.unlock)
    return repo