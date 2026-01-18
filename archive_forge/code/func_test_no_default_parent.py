import sys
import breezy.errors
from breezy import urlutils
from breezy.osutils import getcwd
from breezy.tests import TestCaseWithTransport, TestNotApplicable, TestSkipped
def test_no_default_parent(self):
    """Branches should have no parent by default"""
    b = self.make_branch('.')
    self.assertEqual(None, b.get_parent())