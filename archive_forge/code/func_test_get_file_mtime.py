import time
from breezy import transport
from breezy.tests.per_tree import TestCaseWithTree
def test_get_file_mtime(self):
    now = time.time()
    tree = self.get_basic_tree()
    tree.lock_read()
    self.addCleanup(tree.unlock)
    mtime_file_id = tree.get_file_mtime('one')
    self.assertIsInstance(mtime_file_id, (float, int))
    self.assertTrue(now - 10 * 60 < mtime_file_id < now + 10 + 60, 'now: {:f}, mtime_file_id: {:f}'.format(now, mtime_file_id))
    mtime_path = tree.get_file_mtime('one')
    self.assertEqual(mtime_file_id, mtime_path)