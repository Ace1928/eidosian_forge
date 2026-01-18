import datetime
import math
import re
def format_rfc3339(date_time):
    if date_time.tzinfo is None:
        date_time = date_time.replace(tzinfo=UTC)
    date_time = date_time.astimezone(UTC)
    return date_time.strftime('%Y-%m-%dT%H:%M:%SZ')