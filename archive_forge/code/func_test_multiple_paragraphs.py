import unittest
import commonmark
def test_multiple_paragraphs(self):
    src_markdown = '\nStart of first paragraph that\ncontinues on a new line\n\nThis is the second paragraph\n'
    expected_rst = '\nStart of first paragraph that\ncontinues on a new line\n\nThis is the second paragraph\n'
    self.assertEqualRender(src_markdown, expected_rst)