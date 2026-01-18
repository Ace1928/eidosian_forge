from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
import struct
import sys
import textwrap
import six
from six.moves import range  # pylint: disable=redefined-builtin
def str_or_unicode(value):
    """Converts a value to a python string.

  Behavior of this function is intentionally different in Python2/3.

  In Python2, the given value is attempted to convert to a str (byte string).
  If it contains non-ASCII characters, it is converted to a unicode instead.

  In Python3, the given value is always converted to a str (unicode string).

  This behavior reflects the (bad) practice in Python2 to try to represent
  a string as str as long as it contains ASCII characters only.

  Args:
    value: An object to be converted to a string.

  Returns:
    A string representation of the given value. See the description above
    for its type.
  """
    try:
        return str(value)
    except UnicodeEncodeError:
        return unicode(value)