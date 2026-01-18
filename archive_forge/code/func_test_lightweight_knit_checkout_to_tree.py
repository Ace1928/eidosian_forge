from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_lightweight_knit_checkout_to_tree(self):
    self.test_lightweight_format_checkout_to_tree('knit')