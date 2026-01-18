import io
import sys
from typing import Any, List, Optional, Tuple
import dns.exception
import dns.name
import dns.ttl
def get_uint48(self, base: int=10) -> int:
    """Read the next token and interpret it as a 48-bit unsigned
        integer.

        Raises dns.exception.SyntaxError if not a 48-bit unsigned integer.

        Returns an int.
        """
    value = self.get_int(base=base)
    if value < 0 or value > 281474976710655:
        raise dns.exception.SyntaxError('%d is not an unsigned 48-bit integer' % value)
    return value