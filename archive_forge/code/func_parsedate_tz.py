import time, calendar
def parsedate_tz(data):
    """Convert a date string to a time tuple.

    Accounts for military timezones.
    """
    res = _parsedate_tz(data)
    if not res:
        return
    if res[9] is None:
        res[9] = 0
    return tuple(res)