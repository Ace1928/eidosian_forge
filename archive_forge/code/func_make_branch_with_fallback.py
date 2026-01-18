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
def make_branch_with_fallback(self):
    fallback = self.make_branch('fallback')
    if not fallback._format.supports_stacking():
        raise tests.TestNotApplicable('format does not support stacking')
    stacked = self.make_branch('stacked')
    stacked.set_stacked_on_url(fallback.base)
    return stacked