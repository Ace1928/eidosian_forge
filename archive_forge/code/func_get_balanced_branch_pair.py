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
def get_balanced_branch_pair(self):
    """Returns br_a, br_b as with one commit in a, and b has a's stores."""
    tree_a, tree_b = self.get_unbalanced_tree_pair()
    tree_b.branch.repository.fetch(tree_a.branch.repository)
    return (tree_a, tree_b)