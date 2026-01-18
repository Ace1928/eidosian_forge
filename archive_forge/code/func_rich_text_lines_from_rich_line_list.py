import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def rich_text_lines_from_rich_line_list(rich_text_list, annotations=None):
    """Convert a list of RichLine objects or strings to a RichTextLines object.

  Args:
    rich_text_list: a list of RichLine objects or strings
    annotations: annotations for the resultant RichTextLines object.

  Returns:
    A corresponding RichTextLines object.
  """
    lines = []
    font_attr_segs = {}
    for i, rl in enumerate(rich_text_list):
        if isinstance(rl, RichLine):
            lines.append(rl.text)
            if rl.font_attr_segs:
                font_attr_segs[i] = rl.font_attr_segs
        else:
            lines.append(rl)
    return RichTextLines(lines, font_attr_segs, annotations=annotations)