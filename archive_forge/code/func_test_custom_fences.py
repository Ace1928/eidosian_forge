from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import markup
def test_custom_fences(self):
    self.assert_item_types('      ###\n      this is some\n         verbatim text\n      that should not\n         be changed\n      ###\n      ', [markup.CommentType.PARAGRAPH, markup.CommentType.SEPARATOR])
    config = MockRootConfig(fence_pattern='^\\s*([#`~]{3}[#`~]*)(.*)$', ruler_pattern='^\\s*[^\\w\\s]{3}.*[^\\w\\s]{3}$')
    self.assert_item_types('      ###\n      this is some\n         verbatim text\n      that should not\n         be changed\n      ###\n      ', [markup.CommentType.FENCE, markup.CommentType.VERBATIM, markup.CommentType.FENCE, markup.CommentType.SEPARATOR], config=config)