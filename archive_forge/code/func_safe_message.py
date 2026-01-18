from __future__ import (absolute_import, division, print_function)
import sys
def safe_message(value):
    """Given an input value as text or bytes, return the first non-empty line as text, ensuring it can be round-tripped as UTF-8."""
    if isinstance(value, Text):
        value = value.encode(ENCODING, ERRORS)
    value = value.decode(ENCODING, ERRORS)
    value = value.strip().splitlines()[0].strip()
    return value