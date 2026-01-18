from __future__ import absolute_import
import six
import copy
from collections import OrderedDict
from googleapiclient import _helpers as util
def emitEnd(self, text, comment):
    """Add text and comment to the output with line terminator.

    Args:
      text: string, Text to output.
      comment: string, Python comment.
    """
    if comment:
        divider = '\n' + '  ' * (self.dent + 2) + '# '
        lines = comment.splitlines()
        lines = [x.rstrip() for x in lines]
        comment = divider.join(lines)
        self.value.extend([text, ' # ', comment, '\n'])
    else:
        self.value.extend([text, '\n'])