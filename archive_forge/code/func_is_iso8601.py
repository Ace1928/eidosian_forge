import datetime
import re
import typing
from decimal import Decimal
def is_iso8601(datestring: str) -> bool:
    """Check if a string matches an ISO 8601 format.

    :param datestring: The string to check for validity
    :returns: True if the string matches an ISO 8601 format, False otherwise
    """
    try:
        m = ISO8601_REGEX.match(datestring)
        return bool(m)
    except Exception as e:
        raise ParseError(e)