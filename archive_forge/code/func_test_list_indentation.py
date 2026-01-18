from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import markup
def test_list_indentation(self):
    lines = [line[6:] for line in '      * this is a\n      * bulleted list\n      * of three items\n\n        * this is another list\n        * of two items\n\n          * this is a third list\n          * of two items\n    '.splitlines()]
    items = markup.parse(lines)
    self.assertEqual(items[0].kind, markup.CommentType.BULLET_LIST)
    self.assertEqual(items[0].indent, 0)
    self.assertEqual(items[2].kind, markup.CommentType.BULLET_LIST)
    self.assertEqual(items[2].indent, 2)
    self.assertEqual(items[4].kind, markup.CommentType.BULLET_LIST)
    self.assertEqual(items[4].indent, 4)