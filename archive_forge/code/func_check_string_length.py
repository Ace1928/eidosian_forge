import collections.abc
import math
import re
import unicodedata
import urllib
from oslo_utils._i18n import _
from oslo_utils import encodeutils
def check_string_length(value, name=None, min_length=0, max_length=None):
    """Check the length of specified string.

    :param value: the value of the string
    :param name: the name of the string
    :param min_length: the min_length of the string
    :param max_length: the max_length of the string
    :raises TypeError, ValueError: For any invalid input.

    .. versionadded:: 3.7
    """
    if name is None:
        name = value
    if not isinstance(value, str):
        msg = _('%s is not a string or unicode') % name
        raise TypeError(msg)
    length = len(value)
    if length < min_length:
        msg = _('%(name)s has %(length)s characters, less than %(min_length)s.') % {'name': name, 'length': length, 'min_length': min_length}
        raise ValueError(msg)
    if max_length and length > max_length:
        msg = _('%(name)s has %(length)s characters, more than %(max_length)s.') % {'name': name, 'length': length, 'max_length': max_length}
        raise ValueError(msg)