import collections
import os
import re
import sys
from datetime import timedelta
from pyudev._errors import (
from pyudev._util import (
def asint(self, attribute):
    """
        Get the given ``attribute`` as an int.

        :param attribute: the key for an attribute value
        :type attribute: unicode or byte string
        :returns: the value corresponding to ``attribute``, as an int
        :rtype: int
        :raises KeyError: if no value found for ``attribute``
        :raises UnicodeDecodeError: if value is not convertible to unicode
        :raises ValueError: if unicode value can not be converted to an int
        """
    return int(self.asstring(attribute))