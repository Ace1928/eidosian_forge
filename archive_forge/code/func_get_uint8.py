from io import StringIO
import sys
import dns.exception
import dns.name
import dns.ttl
from ._compat import long, text_type, binary_type
def get_uint8(self):
    """Read the next token and interpret it as an 8-bit unsigned
        integer.

        Raises dns.exception.SyntaxError if not an 8-bit unsigned integer.

        Returns an int.
        """
    value = self.get_int()
    if value < 0 or value > 255:
        raise dns.exception.SyntaxError('%d is not an unsigned 8-bit integer' % value)
    return value