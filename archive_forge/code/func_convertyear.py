from __future__ import unicode_literals
import datetime
import re
import string
import time
import warnings
from calendar import monthrange
from io import StringIO
import six
from six import integer_types, text_type
from decimal import Decimal
from warnings import warn
from .. import relativedelta
from .. import tz
def convertyear(self, year, century_specified=False):
    """
        Converts two-digit years to year within [-50, 49]
        range of self._year (current local time)
        """
    assert year >= 0
    if year < 100 and (not century_specified):
        year += self._century
        if year >= self._year + 50:
            year -= 100
        elif year < self._year - 50:
            year += 100
    return year