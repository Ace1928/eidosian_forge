import datetime
import struct
from six.moves import winreg
from six import text_type
from ._common import tzrangebase
def picknthweekday(year, month, dayofweek, hour, minute, whichweek):
    """ dayofweek == 0 means Sunday, whichweek 5 means last instance """
    first = datetime.datetime(year, month, 1, hour, minute)
    weekdayone = first.replace(day=(dayofweek - first.isoweekday()) % 7 + 1)
    wd = weekdayone + (whichweek - 1) * ONEWEEK
    if wd.month != month:
        wd -= ONEWEEK
    return wd