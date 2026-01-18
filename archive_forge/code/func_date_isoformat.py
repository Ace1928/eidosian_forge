import re
from datetime import date, timedelta
from isodate.isostrf import strftime, DATE_EXT_COMPLETE
from isodate.isoerror import ISO8601Error
def date_isoformat(tdate, format=DATE_EXT_COMPLETE, yeardigits=4):
    """
    Format date strings.

    This method is just a wrapper around isodate.isostrf.strftime and uses
    Date-Extended-Complete as default format.
    """
    return strftime(tdate, format, yeardigits)