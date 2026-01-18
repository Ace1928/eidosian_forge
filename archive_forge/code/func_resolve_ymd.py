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
def resolve_ymd(self, yearfirst, dayfirst):
    len_ymd = len(self)
    year, month, day = (None, None, None)
    strids = (('y', self.ystridx), ('m', self.mstridx), ('d', self.dstridx))
    strids = {key: val for key, val in strids if val is not None}
    if len(self) == len(strids) > 0 or (len(self) == 3 and len(strids) == 2):
        return self._resolve_from_stridxs(strids)
    mstridx = self.mstridx
    if len_ymd > 3:
        raise ValueError('More than three YMD values')
    elif len_ymd == 1 or (mstridx is not None and len_ymd == 2):
        if mstridx is not None:
            month = self[mstridx]
            other = self[mstridx - 1]
        else:
            other = self[0]
        if len_ymd > 1 or mstridx is None:
            if other > 31:
                year = other
            else:
                day = other
    elif len_ymd == 2:
        if self[0] > 31:
            year, month = self
        elif self[1] > 31:
            month, year = self
        elif dayfirst and self[1] <= 12:
            day, month = self
        else:
            month, day = self
    elif len_ymd == 3:
        if mstridx == 0:
            if self[1] > 31:
                month, year, day = self
            else:
                month, day, year = self
        elif mstridx == 1:
            if self[0] > 31 or (yearfirst and self[2] <= 31):
                year, month, day = self
            else:
                day, month, year = self
        elif mstridx == 2:
            if self[1] > 31:
                day, year, month = self
            else:
                year, day, month = self
        elif self[0] > 31 or self.ystridx == 0 or (yearfirst and self[1] <= 12 and (self[2] <= 31)):
            if dayfirst and self[2] <= 12:
                year, day, month = self
            else:
                year, month, day = self
        elif self[0] > 12 or (dayfirst and self[1] <= 12):
            day, month, year = self
        else:
            month, day, year = self
    return (year, month, day)