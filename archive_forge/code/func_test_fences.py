from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import markup
def test_fences(self):
    self.assert_item_types('      ~~~\n      this is some\n         verbatim text\n      that should not\n         be changed\n      ~~~~~~\n      ', [markup.CommentType.FENCE, markup.CommentType.VERBATIM, markup.CommentType.FENCE, markup.CommentType.SEPARATOR])