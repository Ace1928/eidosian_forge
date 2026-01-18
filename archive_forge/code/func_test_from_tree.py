import os
from breezy import osutils
from breezy.bzr.testament import StrictTestament, StrictTestament3, Testament
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import SymlinkFeature
def test_from_tree(self):
    tree = self.b.repository.revision_tree(b'test@user-2')
    testament = self.testament_class().from_revision_tree(tree)
    text_1 = testament.as_short_text()
    text_2 = self.from_revision(self.b.repository, b'test@user-2').as_short_text()
    self.assertEqual(text_1, text_2)