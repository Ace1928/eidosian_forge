from breezy import revision
from breezy.tests import TestCaseWithTransport
from breezy.tree import FileTimestampUnavailable
def test_get_file_revision_root(self):
    self.assertEqual(self.rev_id, self.rev_tree.get_file_revision(''))