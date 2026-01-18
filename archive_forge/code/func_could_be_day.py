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
def could_be_day(self, value):
    if self.has_day:
        return False
    elif not self.has_month:
        return 1 <= value <= 31
    elif not self.has_year:
        month = self[self.mstridx]
        return 1 <= value <= monthrange(2000, month)[1]
    else:
        month = self[self.mstridx]
        year = self[self.ystridx]
        return 1 <= value <= monthrange(year, month)[1]