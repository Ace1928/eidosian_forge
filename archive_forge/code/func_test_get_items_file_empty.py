import sys
from breezy import rules, tests
def test_get_items_file_empty(self):
    rs = self.make_searcher('')
    self.assertEqual((), rs.get_items('a.txt'))
    self.assertEqual((), rs.get_selected_items('a.txt', ['foo']))
    self.assertEqual(None, rs.get_single_value('a.txt', 'foo'))