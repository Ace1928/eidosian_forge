import struct
import dns.exception
from ._compat import binary_type
def inet_aton(text):
    """Convert an IPv4 address in text form to binary form.

    *text*, a ``text``, the IPv4 address in textual form.

    Returns a ``binary``.
    """
    if not isinstance(text, binary_type):
        text = text.encode()
    parts = text.split(b'.')
    if len(parts) != 4:
        raise dns.exception.SyntaxError
    for part in parts:
        if not part.isdigit():
            raise dns.exception.SyntaxError
        if len(part) > 1 and part[0] == '0':
            raise dns.exception.SyntaxError
    try:
        bytes = [int(part) for part in parts]
        return struct.pack('BBBB', *bytes)
    except:
        raise dns.exception.SyntaxError