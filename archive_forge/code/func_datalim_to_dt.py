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
def datalim_to_dt(self):
    """Convert axis data interval to datetime objects."""
    dmin, dmax = self.axis.get_data_interval()
    if dmin > dmax:
        dmin, dmax = (dmax, dmin)
    return (num2date(dmin, self.tz), num2date(dmax, self.tz))