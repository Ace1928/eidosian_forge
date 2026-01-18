from breezy import revision
from breezy.tests import TestCaseWithTransport
from breezy.tree import FileTimestampUnavailable
def test_empty_no_unknowns(self):
    self.assertEqual([], list(self.rev_tree.unknowns()))