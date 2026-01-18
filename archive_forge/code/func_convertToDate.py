import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import collections
import pprint
import traceback
import types
from datetime import datetime
@staticmethod
def convertToDate(fmt='%Y-%m-%d'):
    """
        Helper to create a parse action for converting parsed date string to Python datetime.date

        Params -
         - fmt - format to be passed to datetime.strptime (default=C{"%Y-%m-%d"})

        Example::
            date_expr = pyparsing_common.iso8601_date.copy()
            date_expr.setParseAction(pyparsing_common.convertToDate())
            print(date_expr.parseString("1999-12-31"))
        prints::
            [datetime.date(1999, 12, 31)]
        """

    def cvt_fn(s, l, t):
        try:
            return datetime.strptime(t[0], fmt).date()
        except ValueError as ve:
            raise ParseException(s, l, str(ve))
    return cvt_fn