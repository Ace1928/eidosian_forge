import os
from breezy.tests import TestCaseWithTransport
def test_many_added(self):
    with self.tracker:
        self.build_tree_contents([('tree/f%d' % d, 'content') for d in range(100)])
        self.assertTrue(self.tracker.is_dirty())
        self.assertEqual(self.tracker.relpaths(), {'f%d' % d for d in range(100)})