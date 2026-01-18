import os
from breezy.tests import TestCaseWithTransport
def test_added_then_deleted(self):
    with self.tracker:
        self.tracker.mark_clean()
        self.assertFalse(self.tracker.is_dirty())
        self.build_tree_contents([('tree/foo', 'bar')])
        os.unlink('tree/foo')
        self.assertFalse(self.tracker.is_dirty())
        self.assertEqual(self.tracker.relpaths(), set())