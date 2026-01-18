from __future__ import unicode_literals
import math
import textwrap
import re
from cmakelang import common
def is_hashruler(item):
    """
  Return true if the markup item is a hash ruler, i.e.::

      ###########################
      # Like this ^^^ or this vvv
      ###########################
  """
    if item.kind != CommentType.RULER:
        return False
    if len(item.lines) != 1:
        return False
    if item.lines[0].strip('#'):
        return False
    return True