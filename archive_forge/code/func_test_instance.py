from ...controldir import format_registry
from ...repository import InterRepository
from ...tests import TestCaseWithTransport
from ..interrepo import InterToGitRepository
from ..mapping import BzrGitMappingExperimental, BzrGitMappingv1
def test_instance(self):
    self.assertIsInstance(self._get_interrepo(), InterToGitRepository)