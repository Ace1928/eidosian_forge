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
def test_generate_revision_history(self):
    """Create a fake revision history easily."""
    tree = self.make_branch_and_tree('.')
    rev1 = tree.commit('foo')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    graph = tree.branch.repository.get_graph()
    orig_history = list(graph.iter_lefthand_ancestry(tree.branch.last_revision(), [revision.NULL_REVISION]))
    rev2 = tree.commit('bar', allow_pointless=True)
    tree.branch.generate_revision_history(rev1)
    self.assertEqual(orig_history, list(graph.iter_lefthand_ancestry(tree.branch.last_revision(), [revision.NULL_REVISION])))