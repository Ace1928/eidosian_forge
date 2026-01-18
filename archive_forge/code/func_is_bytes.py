from __future__ import absolute_import
import itertools
import sys
from struct import pack
def is_bytes(obj):
    """
    Determines whether the given value is a byte string.

    :param obj:
        The value to test.
    :returns:
        ``True`` if ``value`` is a byte string; ``False`` otherwise.
    """
    return isinstance(obj, bytes)