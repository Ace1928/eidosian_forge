import os
from breezy.tests import TestCaseWithTransport
def test_directory_added(self):
    with self.tracker:
        self.build_tree_contents([('tree/foo/',)])
        self.assertTrue(self.tracker.is_dirty())
        self.assertEqual(self.tracker.relpaths(), {'foo'})