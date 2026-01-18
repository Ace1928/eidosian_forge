from datetime import (
import time
import unittest
def format_microsecond(timestamp, utc=False, use_system_timezone=True):
    """
    Same as `rfc3339.format` but with the microsecond fraction after the seconds.
    """
    return _format(timestamp, _string_microseconds, utc, use_system_timezone)