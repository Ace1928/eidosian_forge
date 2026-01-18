import re
from datetime import tzinfo, datetime, timedelta
def parse_date_allow_empty(datestring, default_timezone=UTC):
    """
    Parses ISO 8601 dates into datetime objects, but allow empty values.

    In case empty value is found, None is returned.
    """
    return parse_date(datestring=datestring, default_timezone=default_timezone, allow_empty=True)