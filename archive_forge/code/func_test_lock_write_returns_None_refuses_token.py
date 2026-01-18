from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
def test_lock_write_returns_None_refuses_token(self):
    branch = self.make_branch('b')
    with branch.lock_write() as lock:
        if lock.token is not None:
            return
        self.assertRaises(errors.TokenLockingNotSupported, branch.lock_write, token='token')