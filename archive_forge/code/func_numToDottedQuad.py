import re
import sys
from pprint import pprint
def numToDottedQuad(num):
    """
    Convert int or long int to dotted quad string
    
    >>> numToDottedQuad(long(-1))
    Traceback (most recent call last):
    ValueError: Not a good numeric IP: -1
    >>> numToDottedQuad(long(1))
    '0.0.0.1'
    >>> numToDottedQuad(long(16777218))
    '1.0.0.2'
    >>> numToDottedQuad(long(16908291))
    '1.2.0.3'
    >>> numToDottedQuad(long(16909060))
    '1.2.3.4'
    >>> numToDottedQuad(long(4294967295))
    '255.255.255.255'
    >>> numToDottedQuad(long(4294967296))
    Traceback (most recent call last):
    ValueError: Not a good numeric IP: 4294967296
    >>> numToDottedQuad(-1)
    Traceback (most recent call last):
    ValueError: Not a good numeric IP: -1
    >>> numToDottedQuad(1)
    '0.0.0.1'
    >>> numToDottedQuad(16777218)
    '1.0.0.2'
    >>> numToDottedQuad(16908291)
    '1.2.0.3'
    >>> numToDottedQuad(16909060)
    '1.2.3.4'
    >>> numToDottedQuad(4294967295)
    '255.255.255.255'
    >>> numToDottedQuad(4294967296)
    Traceback (most recent call last):
    ValueError: Not a good numeric IP: 4294967296

    """
    import socket, struct
    if num > long(4294967295) or num < 0:
        raise ValueError('Not a good numeric IP: %s' % num)
    try:
        return socket.inet_ntoa(struct.pack('!L', long(num)))
    except (socket.error, struct.error, OverflowError):
        raise ValueError('Not a good numeric IP: %s' % num)