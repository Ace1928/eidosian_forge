import unittest
import commonmark
def assertEqualRender(self, src_markdown, expected_rst):
    rendered_rst = self.render_rst(src_markdown)
    self.assertEqual(rendered_rst, expected_rst)