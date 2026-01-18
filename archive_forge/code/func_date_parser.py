from statsmodels.compat.python import asstr, lmap, lrange, lzip
import datetime
import re
import numpy as np
from pandas import to_datetime
def date_parser(timestr, parserinfo=None, **kwargs):
    """
    Uses dateutil.parser.parse, but also handles monthly dates of the form
    1999m4, 1999:m4, 1999:mIV, 1999mIV and the same for quarterly data
    with q instead of m. It is not case sensitive. The default for annual
    data is the end of the year, which also differs from dateutil.
    """
    flags = re.IGNORECASE | re.VERBOSE
    if re.search(_q_pattern, timestr, flags):
        y, q = timestr.replace(':', '').lower().split('q')
        month, day = _quarter_to_day[q.upper()]
        year = int(y)
    elif re.search(_m_pattern, timestr, flags):
        y, m = timestr.replace(':', '').lower().split('m')
        month, day = _month_to_day[m.upper()]
        year = int(y)
        if _is_leap(y) and month == 2:
            day += 1
    elif re.search(_y_pattern, timestr, flags):
        month, day = (12, 31)
        year = int(timestr)
    else:
        return to_datetime(timestr, **kwargs)
    return datetime.datetime(year, month, day)