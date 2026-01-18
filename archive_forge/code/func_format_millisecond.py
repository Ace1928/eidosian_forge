from datetime import (
import time
import unittest
def format_millisecond(timestamp, utc=False, use_system_timezone=True):
    """
    Same as `rfc3339.format` but with the millisecond fraction after the seconds.
    """
    return _format(timestamp, _string_milliseconds, utc, use_system_timezone)