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
def test_find_branches_using(self):
    try:
        repo = self.make_repository_and_foo_bar(shared=True)
    except errors.IncompatibleFormat:
        raise tests.TestNotApplicable
    branches = list(repo.find_branches(using=True))
    self.assertContainsRe(branches[-1].base, 'repository/foo/$')
    if len(branches) == 2:
        self.assertContainsRe(branches[-2].base, 'repository/$')
    else:
        self.assertEqual(1, len(branches))