from io import StringIO
import sys
import dns.exception
import dns.name
import dns.ttl
from ._compat import long, text_type, binary_type
def get_uint16(self, base=10):
    """Read the next token and interpret it as a 16-bit unsigned
        integer.

        Raises dns.exception.SyntaxError if not a 16-bit unsigned integer.

        Returns an int.
        """
    value = self.get_int(base=base)
    if value < 0 or value > 65535:
        if base == 8:
            raise dns.exception.SyntaxError('%o is not an octal unsigned 16-bit integer' % value)
        else:
            raise dns.exception.SyntaxError('%d is not an unsigned 16-bit integer' % value)
    return value