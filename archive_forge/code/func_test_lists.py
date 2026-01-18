from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import markup
def test_lists(self):
    self.assert_item_types('      This is a paragraph\n\n      * this is a\n      * bulleted list\n      * of three items\n\n        1. this is another list\n        2. of two items\n\n      This is a paragraph', [markup.CommentType.PARAGRAPH, markup.CommentType.SEPARATOR, markup.CommentType.BULLET_LIST, markup.CommentType.SEPARATOR, markup.CommentType.ENUM_LIST, markup.CommentType.SEPARATOR, markup.CommentType.PARAGRAPH])