import calendar
import datetime
import logging
import os
import time
import warnings
from tzlocal import windows_tz
def assert_tz_offset(tz, error=True):
    """Assert that system's timezone offset equals to the timezone offset found.

    If they don't match, we probably have a misconfiguration, for example, an
    incorrect timezone set in /etc/timezone file in systemd distributions.

    If error is True, this method will raise a ValueError, otherwise it will
    emit a warning.
    """
    tz_offset = get_tz_offset(tz)
    system_offset = calendar.timegm(time.localtime()) - calendar.timegm(time.gmtime())
    if abs(tz_offset - system_offset) > 60:
        msg = f'Timezone offset does not match system offset: {tz_offset} != {system_offset}. Please, check your config files.'
        if error:
            raise ValueError(msg)
        warnings.warn(msg)