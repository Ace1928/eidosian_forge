import re
from io import BytesIO
from ... import branch as _mod_branch
from ... import commit, controldir
from ... import delta as _mod_delta
from ... import errors, gpg, info, repository
from ... import revision as _mod_revision
from ... import tests, transport, upgrade, workingtree
from ...bzr import branch as _mod_bzrbranch
from ...bzr import inventory, knitpack_repo, remote
from ...bzr import repository as bzrrepository
from .. import per_repository, test_server
from ..matchers import *
def make_repository_and_foo_bar(self, shared=None):
    made_control = self.make_controldir('repository')
    repo = made_control.create_repository(shared=shared)
    if not repo._format.supports_nesting_repositories:
        raise tests.TestNotApplicable('repository does not support nesting repositories')
    controldir.ControlDir.create_branch_convenience(self.get_url('repository/foo'), force_new_repo=False)
    controldir.ControlDir.create_branch_convenience(self.get_url('repository/bar'), force_new_repo=True)
    baz = self.make_controldir('repository/baz')
    qux = self.make_branch('repository/baz/qux')
    quxx = self.make_branch('repository/baz/qux/quxx')
    return repo