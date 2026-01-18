from breezy import errors, urlutils
from breezy.bzr import remote
from breezy.controldir import ControlDir
from breezy.tests import multiply_tests
from breezy.tests.per_repository import (TestCaseWithRepository,
def readonly_repository(self, repo):
    relpath = urlutils.basename(repo.controldir.user_url.rstrip('/'))
    return ControlDir.open_from_transport(self.get_readonly_transport(relpath)).open_repository()