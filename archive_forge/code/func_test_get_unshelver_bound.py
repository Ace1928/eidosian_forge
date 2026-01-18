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
def test_get_unshelver_bound(self):
    tree = self.make_branch_and_tree('tree')
    tree.commit('')
    self.build_tree_contents([('tree/file', b'contents1')])
    tree.add('file')
    with skip_if_storing_uncommitted_unsupported():
        tree.store_uncommitted()
    branch = self.make_branch('branch')
    self.bind(branch, tree.branch)
    unshelver = branch.get_unshelver(tree)
    self.assertIsNot(None, unshelver)