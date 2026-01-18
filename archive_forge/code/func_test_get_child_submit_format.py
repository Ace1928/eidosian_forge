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
def test_get_child_submit_format(self):
    branch = self.get_branch()
    branch.get_config_stack().set('child_submit_format', '10')
    branch = self.get_branch()
    self.assertEqual('10', branch.get_child_submit_format())