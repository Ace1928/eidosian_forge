import contextlib
from breezy import branch as _mod_branch
from breezy import config, controldir
from breezy import delta as _mod_delta
from breezy import (errors, lock, merge, osutils, repository, revision, shelf,
from breezy import tree as _mod_tree
from breezy import urlutils
from breezy.bzr import remote
from breezy.tests import per_branch
from breezy.tests.http_server import HttpServer
from breezy.transport import memory
def test_clone_partial(self):
    """Copy only part of the history of a branch."""
    wt_a = self.make_branch_and_tree('a')
    self.build_tree(['a/one'])
    wt_a.add(['one'])
    rev1 = wt_a.commit('commit one')
    self.build_tree(['a/two'])
    wt_a.add(['two'])
    wt_a.commit('commit two')
    repo_b = self.make_repository('b')
    wt_a.branch.repository.copy_content_into(repo_b)
    branch = wt_a.branch.controldir.open_branch()
    br_b = branch.clone(repo_b.controldir, revision_id=rev1)
    self.assertEqual(rev1, br_b.last_revision())