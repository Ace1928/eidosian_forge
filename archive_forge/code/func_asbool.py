import collections
import os
import re
import sys
from datetime import timedelta
from pyudev._errors import (
from pyudev._util import (
def asbool(self, attribute):
    """
        Get the given ``attribute`` from this device as a bool.

        :param attribute: the key for an attribute value
        :type attribute: unicode or byte string
        :returns: the value corresponding to ``attribute``, as bool
        :rtype: bool
        :raises KeyError: if no value found for ``attribute``
        :raises UnicodeDecodeError: if value is not convertible to unicode
        :raises ValueError: if unicode value can not be converted to a bool

        A boolean attribute has either a value of ``'1'`` or of ``'0'``,
        where ``'1'`` stands for ``True``, and ``'0'`` for ``False``.  Any
        other value causes a :exc:`~exceptions.ValueError` to be raised.
        """
    return string_to_bool(self.asstring(attribute))