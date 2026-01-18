import unittest
import time
from datetime import datetime, timedelta
from isodate import strftime
from isodate import LOCAL
from isodate import DT_EXT_COMPLETE
from isodate import tzinfo
def localtime_mock(self, secs):
    """
            mock time.localtime so that it always returns a time_struct with
            tm_idst=1
            """
    tt = self.ORIG['localtime'](secs)
    if tt.tm_year < 2000:
        dst = 1
    else:
        dst = 0
    tt = (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec, tt.tm_wday, tt.tm_yday, dst)
    return time.struct_time(tt)