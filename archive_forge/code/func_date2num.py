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
def date2num(d):
    """
    Convert datetime objects to Matplotlib dates.

    Parameters
    ----------
    d : `datetime.datetime` or `numpy.datetime64` or sequences of these

    Returns
    -------
    float or sequence of floats
        Number of days since the epoch.  See `.get_epoch` for the
        epoch, which can be changed by :rc:`date.epoch` or `.set_epoch`.  If
        the epoch is "1970-01-01T00:00:00" (default) then noon Jan 1 1970
        ("1970-01-01T12:00:00") returns 0.5.

    Notes
    -----
    The Gregorian calendar is assumed; this is not universal practice.
    For details see the module docstring.
    """
    d = cbook._unpack_to_numpy(d)
    iterable = np.iterable(d)
    if not iterable:
        d = [d]
    masked = np.ma.is_masked(d)
    mask = np.ma.getmask(d)
    d = np.asarray(d)
    if not np.issubdtype(d.dtype, np.datetime64):
        if not d.size:
            return d
        tzi = getattr(d[0], 'tzinfo', None)
        if tzi is not None:
            d = [dt.astimezone(UTC).replace(tzinfo=None) for dt in d]
            d = np.asarray(d)
        d = d.astype('datetime64[us]')
    d = np.ma.masked_array(d, mask=mask) if masked else d
    d = _dt64_to_ordinalf(d)
    return d if iterable else d[0]