import sys as _sys
import struct as _struct
from socket import inet_aton as _inet_aton
from netaddr.core import AddrFormatError, ZEROFILL, INET_ATON, INET_PTON
from netaddr.strategy import (
def expand_partial_address(addr):
    """
    Expands a partial IPv4 address into a full 4-octet version.

    :param addr: an partial or abbreviated IPv4 address

    :return: an expanded IP address in presentation format (x.x.x.x)

    >>> expand_partial_address('1.2')
    '1.2.0.0'

    .. versionadded:: 1.1.0
    """
    tokens = []
    error = AddrFormatError('invalid partial IPv4 address: %r!' % addr)
    if isinstance(addr, str):
        if ':' in addr:
            raise error
        try:
            if '.' in addr:
                tokens = ['%d' % int(o) for o in addr.split('.')]
            else:
                tokens = ['%d' % int(addr)]
        except ValueError:
            raise error
        if 1 <= len(tokens) <= 4:
            for i in range(4 - len(tokens)):
                tokens.append('0')
        else:
            raise error
    if not tokens:
        raise error
    return '%s.%s.%s.%s' % tuple(tokens)