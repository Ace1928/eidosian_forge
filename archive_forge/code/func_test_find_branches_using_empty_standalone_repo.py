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
def test_find_branches_using_empty_standalone_repo(self):
    try:
        repo = self.make_repository('repo', shared=False)
    except errors.IncompatibleFormat:
        raise tests.TestNotApplicable('format does not support standalone repositories')
    try:
        repo.controldir.open_branch()
    except errors.NotBranchError:
        self.assertEqual([], list(repo.find_branches(using=True)))
    else:
        self.assertEqual([repo.controldir.root_transport.base], [b.base for b in repo.find_branches(using=True)])