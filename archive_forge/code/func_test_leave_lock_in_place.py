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
def test_leave_lock_in_place(self):
    repo = self.make_repository('r')
    token = repo.lock_write().repository_token
    try:
        if token is None:
            self.assertRaises(NotImplementedError, repo.leave_lock_in_place)
            return
        repo.leave_lock_in_place()
    finally:
        repo.unlock()
    self.assertRaises(errors.LockContention, repo.lock_write)
    repo.lock_write(token)
    repo.dont_leave_lock_in_place()
    repo.unlock()