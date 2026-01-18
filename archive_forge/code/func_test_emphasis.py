import unittest
import commonmark
def test_emphasis(self):
    src_markdown = 'Hello *Emphasis*'
    expected_rst = '\nHello *Emphasis*\n'
    self.assertEqualRender(src_markdown, expected_rst)