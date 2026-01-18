import unittest
import commonmark
def test_heading(self):
    src_markdown = '\n# Heading 1\n\n## Heading 2\n\n### Heading 3\n\n#### Heading 4\n\n##### Heading 5\n\n###### Heading 6\n'
    expected_rst = '\nHeading 1\n#########\n\nHeading 2\n*********\n\nHeading 3\n=========\n\nHeading 4\n---------\n\nHeading 5\n^^^^^^^^^\n\nHeading 6\n"""""""""\n'
    self.assertEqualRender(src_markdown, expected_rst)