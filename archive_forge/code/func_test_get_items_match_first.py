import sys
from breezy import rules, tests
def test_get_items_match_first(self):
    rs = self.make_searcher('[name ./a.txt]\nfoo=baz\n[name *.txt]\nfoo=bar\na=True\n')
    self.assertEqual((('foo', 'baz'),), rs.get_items('a.txt'))
    self.assertEqual('baz', rs.get_single_value('a.txt', 'foo'))
    self.assertEqual((('foo', 'bar'), ('a', 'True')), rs.get_items('dir/a.txt'))
    self.assertEqual('bar', rs.get_single_value('dir/a.txt', 'foo'))