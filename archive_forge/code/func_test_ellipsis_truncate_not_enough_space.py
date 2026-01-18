from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import formatting
from fire import testutils
def test_ellipsis_truncate_not_enough_space(self):
    text = 'This is a string'
    truncated_text = formatting.EllipsisTruncate(text=text, available_space=2, line_length=LINE_LENGTH)
    self.assertEqual('This is a string', truncated_text)