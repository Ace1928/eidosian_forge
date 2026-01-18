from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def make_repository_with_without_trees(self, with_trees):
    repo = self.make_repository('repo', shared=True)
    repo.set_make_working_trees(with_trees)
    return repo