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
def test_clone_branch_parent(self):
    branch_b = self.get_parented_branch()
    repo_c = self.make_repository('c')
    branch_b.repository.copy_content_into(repo_c)
    branch_c = branch_b.clone(repo_c.controldir)
    self.assertNotEqual(None, branch_c.get_parent())
    self.assertEqual(branch_b.get_parent(), branch_c.get_parent())
    random_parent = 'http://example.com/path/to/branch'
    branch_b.set_parent(random_parent)
    repo_d = self.make_repository('d')
    branch_b.repository.copy_content_into(repo_d)
    branch_d = branch_b.clone(repo_d.controldir)
    self.assertEqual(random_parent, branch_d.get_parent())