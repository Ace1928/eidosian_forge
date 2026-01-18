from __future__ import absolute_import, division, print_function
import sys
def convert_int_to_bytes(no, count=None):
    """
    Convert the absolute value of an integer to a byte string in network byte order.

    If ``count`` is provided, it must be sufficiently large so that the integer's
    absolute value can be represented with these number of bytes. The resulting byte
    string will have length exactly ``count``.

    The value zero will be converted to an empty byte string if ``count`` is provided.
    """
    no = abs(no)
    if count is None:
        count = count_bytes(no)
    return _convert_int_to_bytes(count, no)