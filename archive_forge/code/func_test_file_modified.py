import os
from breezy.tests import TestCaseWithTransport
def test_file_modified(self):
    self.build_tree_contents([('tree/foo', 'bla')])
    with self.tracker:
        self.assertFalse(self.tracker.is_dirty())
        self.build_tree_contents([('tree/foo', 'bar')])
        self.assertTrue(self.tracker.is_dirty())
        self.assertEqual(self.tracker.relpaths(), {'foo'})