import datetime
import functools
import logging
import re
from dateutil.rrule import (rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY,
from dateutil.relativedelta import relativedelta
import dateutil.parser
import dateutil.tz
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, ticker, units
def datestr2num(d, default=None):
    """
    Convert a date string to a datenum using `dateutil.parser.parse`.

    Parameters
    ----------
    d : str or sequence of str
        The dates to convert.

    default : datetime.datetime, optional
        The default date to use when fields are missing in *d*.
    """
    if isinstance(d, str):
        dt = dateutil.parser.parse(d, default=default)
        return date2num(dt)
    else:
        if default is not None:
            d = [date2num(dateutil.parser.parse(s, default=default)) for s in d]
            return np.asarray(d)
        d = np.asarray(d)
        if not d.size:
            return d
        return date2num(_dateutil_parser_parse_np_vectorized(d))