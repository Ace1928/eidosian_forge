import os
from breezy.tests import TestCaseWithTransport
def test_control_file(self):
    with self.tracker:
        self.tree.commit('Some change')
        self.assertFalse(self.tracker.is_dirty())
        self.assertEqual(self.tracker.relpaths(), set())