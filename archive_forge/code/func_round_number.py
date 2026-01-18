import collections
import datetime
import decimal
import numbers
import os
import os.path
import re
import time
from humanfriendly.compat import is_string, monotonic
from humanfriendly.deprecation import define_aliases
from humanfriendly.text import concatenate, format, pluralize, tokenize
def round_number(count, keep_width=False):
    """
    Round a floating point number to two decimal places in a human friendly format.

    :param count: The number to format.
    :param keep_width: :data:`True` if trailing zeros should not be stripped,
                       :data:`False` if they can be stripped.
    :returns: The formatted number as a string. If no decimal places are
              required to represent the number, they will be omitted.

    The main purpose of this function is to be used by functions like
    :func:`format_length()`, :func:`format_size()` and
    :func:`format_timespan()`.

    Here are some examples:

    >>> from humanfriendly import round_number
    >>> round_number(1)
    '1'
    >>> round_number(math.pi)
    '3.14'
    >>> round_number(5.001)
    '5'
    """
    text = '%.2f' % float(count)
    if not keep_width:
        text = re.sub('0+$', '', text)
        text = re.sub('\\.$', '', text)
    return text