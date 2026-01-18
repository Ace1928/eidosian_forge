import sys
from breezy import rules, tests
def test_get_items_from_extension_match(self):
    rs = self.make_searcher('[name *.txt]\nfoo=bar\na=True\n')
    self.assertEqual((), rs.get_items('a.py'))
    self.assertEqual((('foo', 'bar'), ('a', 'True')), rs.get_items('a.txt'))
    self.assertEqual((('foo', 'bar'), ('a', 'True')), rs.get_items('dir/a.txt'))
    self.assertEqual((('foo', 'bar'),), rs.get_selected_items('a.txt', ['foo']))
    self.assertEqual('bar', rs.get_single_value('a.txt', 'foo'))