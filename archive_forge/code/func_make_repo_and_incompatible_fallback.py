from breezy import errors, urlutils
from breezy.bzr import remote
from breezy.controldir import ControlDir
from breezy.tests import multiply_tests
from breezy.tests.per_repository import (TestCaseWithRepository,
def make_repo_and_incompatible_fallback(self):
    referring = self.make_repository('referring')
    if referring._format.supports_chks:
        different_fmt = '1.9'
    else:
        different_fmt = '2a'
    fallback = self.make_repository('fallback', format=different_fmt)
    return (referring, fallback)