from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_lightweight_rich_root_pack_checkout_to_tree(self):
    self.test_lightweight_format_checkout_to_tree('rich-root-pack')