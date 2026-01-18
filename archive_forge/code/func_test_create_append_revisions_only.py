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
def test_create_append_revisions_only(self):
    try:
        repo = self.make_repository('.', shared=True)
    except errors.IncompatibleFormat:
        return
    for val in (True, False):
        try:
            branch = self.branch_format.initialize(repo.controldir, append_revisions_only=True)
        except (errors.UninitializableFormat, errors.UpgradeRequired):
            return
        self.assertEqual(True, branch.get_append_revisions_only())
        repo.controldir.destroy_branch()