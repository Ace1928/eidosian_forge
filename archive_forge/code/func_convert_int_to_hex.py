from __future__ import absolute_import, division, print_function
import sys
def convert_int_to_hex(no, digits=None):
    """
    Convert the absolute value of an integer to a string of hexadecimal digits.

    If ``digits`` is provided, the string will be padded on the left with ``0``s so
    that the returned value has length ``digits``. If ``digits`` is not sufficient,
    the string will be longer.
    """
    no = abs(no)
    value = _to_hex(no)
    if digits is not None and len(value) < digits:
        value = '0' * (digits - len(value)) + value
    return value