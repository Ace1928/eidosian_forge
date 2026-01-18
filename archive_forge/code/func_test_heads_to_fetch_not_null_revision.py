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
def test_heads_to_fetch_not_null_revision(self):
    tree = self.make_branch_and_tree('a')
    must_fetch, should_fetch = tree.branch.heads_to_fetch()
    self.assertFalse(revision.NULL_REVISION in must_fetch)
    self.assertFalse(revision.NULL_REVISION in should_fetch)