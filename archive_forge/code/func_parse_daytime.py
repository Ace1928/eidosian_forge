import datetime
import re
def parse_daytime(daytime):
    daytime = daytime.strip()
    reg = re.compile('^(.*?)\\s+at\\s+(.*)$', flags=re.I)
    match = reg.match(daytime)
    if match:
        day, time = match.groups()
    else:
        day = time = daytime
    try:
        day = parse_day(day)
        if match and day is None:
            raise ValueError
    except ValueError as e:
        raise ValueError("Invalid day while parsing daytime: '%s'" % day) from e
    try:
        time = parse_time(time)
        if match and time is None:
            raise ValueError
    except ValueError as e:
        raise ValueError("Invalid time while parsing daytime: '%s'" % time) from e
    if day is None and time is None:
        return None
    return (day, time)