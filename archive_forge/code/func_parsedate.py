import time, calendar
def parsedate(data):
    """Convert a time string to a time tuple."""
    t = parsedate_tz(data)
    if isinstance(t, tuple):
        return t[:9]
    else:
        return t