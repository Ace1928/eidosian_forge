import time
from . import errors
def parse_now(s, lineno=0):
    """Parse a date from a string.

    The format must be exactly "now".
    See the spec for details.
    """
    return (time.time(), 0)