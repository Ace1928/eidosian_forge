import os
from breezy.tests import TestCaseWithTransport
def test_file_removed(self):
    with self.tracker:
        self.build_tree_contents([('tree/foo', 'foo')])
        self.assertTrue(self.tracker.is_dirty())
        self.tracker.mark_clean()
        self.build_tree_contents([('tree/foo', 'bar')])
        self.assertTrue(self.tracker.is_dirty())
        self.assertEqual(self.tracker.relpaths(), {'foo'})