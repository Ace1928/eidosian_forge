from io import BytesIO
from unittest import TestCase
from ..mailmap import Mailmap, read_mailmap
def test_lookup(self):
    m = Mailmap()
    m.add_entry((b'Jane Doe', b'jane@desktop.(none)'), (None, None))
    m.add_entry((b'Joe R. Developer', b'joe@example.com'), None)
    m.add_entry((None, b'cto@company.xx'), (None, b'cto@coompany.xx'))
    m.add_entry((b'Some Dude', b'some@dude.xx'), (b'nick1', b'bugs@company.xx'))
    m.add_entry((b'Other Author', b'other@author.xx'), (b'nick2', b'bugs@company.xx'))
    m.add_entry((b'Other Author', b'other@author.xx'), (None, b'nick2@company.xx'))
    m.add_entry((b'Santa Claus', b'santa.claus@northpole.xx'), (None, b'me@company.xx'))
    self.assertEqual(b'Jane Doe <jane@desktop.(none)>', m.lookup(b'Jane Doe <jane@desktop.(none)>'))
    self.assertEqual(b'Jane Doe <jane@desktop.(none)>', m.lookup(b'Jane Doe <jane@example.com>'))
    self.assertEqual(b'Jane Doe <jane@desktop.(none)>', m.lookup(b'Jane D. <jane@desktop.(none)>'))
    self.assertEqual(b'Some Dude <some@dude.xx>', m.lookup(b'nick1 <bugs@company.xx>'))
    self.assertEqual(b'CTO <cto@company.xx>', m.lookup(b'CTO <cto@coompany.xx>'))