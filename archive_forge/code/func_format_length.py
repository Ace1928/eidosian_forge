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
def format_length(num_metres, keep_width=False):
    """
    Format a metre count as a human readable length.

    :param num_metres: The length to format in metres (float / integer).
    :param keep_width: :data:`True` if trailing zeros should not be stripped,
                       :data:`False` if they can be stripped.
    :returns: The corresponding human readable length (a string).

    This function supports ranges from nanometres to kilometres.

    Some examples:

    >>> from humanfriendly import format_length
    >>> format_length(0)
    '0 metres'
    >>> format_length(1)
    '1 metre'
    >>> format_length(5)
    '5 metres'
    >>> format_length(1000)
    '1 km'
    >>> format_length(0.004)
    '4 mm'
    """
    for unit in reversed(length_size_units):
        if num_metres >= unit['divider']:
            number = round_number(float(num_metres) / unit['divider'], keep_width=keep_width)
            return pluralize(number, unit['singular'], unit['plural'])
    return pluralize(num_metres, 'metre')