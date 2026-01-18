from breezy import errors
from breezy.tests.per_controldir import TestCaseWithControlDir
def test_check_support_status(self):
    if not self.bzrdir_format.is_supported():
        self.assertRaises((errors.UnsupportedFormatError, errors.UnsupportedVcs), self.bzrdir_format.check_support_status, False)
    else:
        self.bzrdir_format.check_support_status(True)
        self.bzrdir_format.check_support_status(False)