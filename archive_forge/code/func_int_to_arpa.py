import sys as _sys
import struct as _struct
from socket import inet_aton as _inet_aton
from netaddr.core import AddrFormatError, ZEROFILL, INET_ATON, INET_PTON
from netaddr.strategy import (
def int_to_arpa(int_val):
    """
    :param int_val: An unsigned integer.

    :return: The reverse DNS lookup for an IPv4 address in network byte
        order integer form.
    """
    words = ['%d' % i for i in int_to_words(int_val)]
    words.reverse()
    words.extend(['in-addr', 'arpa', ''])
    return '.'.join(words)