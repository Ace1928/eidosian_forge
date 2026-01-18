from __future__ import unicode_literals
import math
import textwrap
import re
from cmakelang import common
def format_items(config, line_width, items):
    """
  Return lines of formatted text for the sequence of items within a comment
  block
  """
    outlines = []
    indent_history = []
    for item in items:
        if item.kind in (CommentType.BULLET_LIST, CommentType.ENUM_LIST):
            while indent_history and indent_history[-1] >= item.indent:
                indent_history.pop(-1)
            indent_history.append(item.indent)
            nindent = 2 * (len(indent_history) - 1)
            ilines = format_item(config, line_width - nindent, item)
            outlines.extend((' ' * nindent + iline for iline in ilines))
        else:
            outlines.extend(format_item(config, line_width, item))
            if item.kind != CommentType.SEPARATOR:
                indent_history = []
    return outlines