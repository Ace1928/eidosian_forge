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
@contextlib.contextmanager
def skip_if_storing_uncommitted_unsupported():
    try:
        yield
    except errors.StoringUncommittedNotSupported:
        raise tests.TestNotApplicable('Cannot store uncommitted changes.')