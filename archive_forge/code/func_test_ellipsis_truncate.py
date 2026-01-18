from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import formatting
from fire import testutils
def test_ellipsis_truncate(self):
    text = 'This is a string'
    truncated_text = formatting.EllipsisTruncate(text=text, available_space=10, line_length=LINE_LENGTH)
    self.assertEqual('This is...', truncated_text)