from breezy import errors
from breezy.tests.per_controldir import TestCaseWithControlDir
def test_is_supported(self):
    self.assertIsInstance(self.bzrdir_format.is_supported(), bool)