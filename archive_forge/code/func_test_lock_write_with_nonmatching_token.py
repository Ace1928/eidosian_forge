from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
def test_lock_write_with_nonmatching_token(self):
    branch = self.make_branch('b')
    with branch.lock_write() as lock:
        if lock.token is None:
            return
        different_branch_token = lock.token + b'xxx'
        new_branch = branch.controldir.open_branch()
        new_branch.repository = branch.repository
        self.assertRaises(errors.TokenMismatch, new_branch.lock_write, token=different_branch_token)