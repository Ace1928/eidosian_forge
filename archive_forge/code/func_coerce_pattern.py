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
def coerce_pattern(value, flags=0):
    """
    Coerce strings to compiled regular expressions.

    :param value: A string containing a regular expression pattern
                  or a compiled regular expression.
    :param flags: The flags used to compile the pattern (an integer).
    :returns: A compiled regular expression.
    :raises: :exc:`~exceptions.ValueError` when `value` isn't a string
             and also isn't a compiled regular expression.
    """
    if is_string(value):
        value = re.compile(value, flags)
    else:
        empty_pattern = re.compile('')
        pattern_type = type(empty_pattern)
        if not isinstance(value, pattern_type):
            msg = 'Failed to coerce value to compiled regular expression! (%r)'
            raise ValueError(format(msg, value))
    return value