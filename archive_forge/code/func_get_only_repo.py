from breezy import errors, tests, urlutils
from breezy.bzr import remote
from breezy.tests.per_repository import TestCaseWithRepository
def get_only_repo(self, tree):
    """Open just the repository used by this tree.

        This returns a read locked Repository object without any stacking
        fallbacks.
        """
    repo = tree.branch.repository.controldir.open_repository()
    repo.lock_read()
    self.addCleanup(repo.unlock)
    return repo