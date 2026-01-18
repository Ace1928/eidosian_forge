import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontInfoOpenTypeHeadCreatedValidator(value):
    """
    Version 2+.
    """
    if not isinstance(value, str):
        return False
    if not len(value) == 19:
        return False
    if value.count(' ') != 1:
        return False
    date, time = value.split(' ')
    if date.count('/') != 2:
        return False
    if time.count(':') != 2:
        return False
    year, month, day = date.split('/')
    if len(year) != 4:
        return False
    if len(month) != 2:
        return False
    if len(day) != 2:
        return False
    try:
        year = int(year)
        month = int(month)
        day = int(day)
    except ValueError:
        return False
    if month < 1 or month > 12:
        return False
    monthMaxDay = calendar.monthrange(year, month)[1]
    if day < 1 or day > monthMaxDay:
        return False
    hour, minute, second = time.split(':')
    if len(hour) != 2:
        return False
    if len(minute) != 2:
        return False
    if len(second) != 2:
        return False
    try:
        hour = int(hour)
        minute = int(minute)
        second = int(second)
    except ValueError:
        return False
    if hour < 0 or hour > 23:
        return False
    if minute < 0 or minute > 59:
        return False
    if second < 0 or second > 59:
        return False
    return True