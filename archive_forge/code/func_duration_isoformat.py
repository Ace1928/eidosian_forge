from datetime import timedelta
from decimal import Decimal
import re
from six import string_types
from isodate.duration import Duration
from isodate.isoerror import ISO8601Error
from isodate.isodatetime import parse_datetime
from isodate.isostrf import strftime, D_DEFAULT
def duration_isoformat(tduration, format=D_DEFAULT):
    """
    Format duration strings.

    This method is just a wrapper around isodate.isostrf.strftime and uses
    P%P (D_DEFAULT) as default format.
    """
    if isinstance(tduration, Duration) and (tduration.years < 0 or tduration.months < 0 or tduration.tdelta < timedelta(0)) or (isinstance(tduration, timedelta) and tduration < timedelta(0)):
        ret = '-'
    else:
        ret = ''
    ret += strftime(tduration, format)
    return ret