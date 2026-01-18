from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
def test_lock_write_with_token_fails_when_unlocked(self):
    branch = self.make_branch('b')
    token = branch.lock_write().token
    branch.unlock()
    if token is None:
        return
    self.assertRaises(errors.TokenMismatch, branch.lock_write, token=token)