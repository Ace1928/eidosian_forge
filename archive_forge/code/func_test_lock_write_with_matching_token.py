from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
def test_lock_write_with_matching_token(self):
    """Test that a branch can be locked with a token, if it is already
        locked by that token."""
    branch = self.make_branch('b')
    with branch.lock_write() as lock:
        if lock.token is None:
            return
        branch.lock_write(token=lock.token)
        branch.unlock()
        new_branch = branch.controldir.open_branch()
        new_branch.repository = branch.repository
        new_branch.lock_write(token=lock.token)
        new_branch.unlock()