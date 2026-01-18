from io import StringIO
import sys
import dns.exception
import dns.name
import dns.ttl
from ._compat import long, text_type, binary_type
def get_uint32(self):
    """Read the next token and interpret it as a 32-bit unsigned
        integer.

        Raises dns.exception.SyntaxError if not a 32-bit unsigned integer.

        Returns an int.
        """
    token = self.get().unescape()
    if not token.is_identifier():
        raise dns.exception.SyntaxError('expecting an identifier')
    if not token.value.isdigit():
        raise dns.exception.SyntaxError('expecting an integer')
    value = long(token.value)
    if value < 0 or value > long(4294967296):
        raise dns.exception.SyntaxError('%d is not an unsigned 32-bit integer' % value)
    return value