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
def test_generate_revision_history_NULL_REVISION(self):
    tree = self.make_branch_and_tree('.')
    rev1 = tree.commit('foo')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    tree.branch.generate_revision_history(revision.NULL_REVISION)
    self.assertEqual(revision.NULL_REVISION, tree.branch.last_revision())