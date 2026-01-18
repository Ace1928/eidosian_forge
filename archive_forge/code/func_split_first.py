from __future__ import absolute_import
import re
from collections import namedtuple
from ..exceptions import LocationParseError
from ..packages import six
def split_first(s, delims):
    """
    .. deprecated:: 1.25

    Given a string and an iterable of delimiters, split on the first found
    delimiter. Return two split parts and the matched delimiter.

    If not found, then the first part is the full input string.

    Example::

        >>> split_first('foo/bar?baz', '?/=')
        ('foo', 'bar?baz', '/')
        >>> split_first('foo/bar?baz', '123')
        ('foo/bar?baz', '', None)

    Scales linearly with number of delims. Not ideal for large number of delims.
    """
    min_idx = None
    min_delim = None
    for d in delims:
        idx = s.find(d)
        if idx < 0:
            continue
        if min_idx is None or idx < min_idx:
            min_idx = idx
            min_delim = d
    if min_idx is None or min_idx < 0:
        return (s, '', None)
    return (s[:min_idx], s[min_idx + 1:], min_delim)