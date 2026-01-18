import time
from . import errors
def parse_tz(tz):
    """Parse a timezone specification in the [+|-]HHMM format.

    :return: the timezone offset in seconds.
    """
    sign_byte = tz[0:1]
    if sign_byte not in (b'+', b'-'):
        raise ValueError(tz)
    sign = {b'+': +1, b'-': -1}[sign_byte]
    hours = int(tz[1:-2])
    minutes = int(tz[-2:])
    return sign * 60 * (60 * hours + minutes)