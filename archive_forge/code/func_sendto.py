from base64 import b64encode
import six
from errno import EOPNOTSUPP, EINVAL, EAGAIN
import functools
from io import BytesIO
import logging
import os
from os import SEEK_CUR
import socket
import struct
import sys
def sendto(self, bytes, *args, **kwargs):
    if self.type != socket.SOCK_DGRAM:
        return super(socksocket, self).sendto(bytes, *args, **kwargs)
    if not self._proxyconn:
        self.bind(('', 0))
    address = args[-1]
    flags = args[:-1]
    header = BytesIO()
    RSV = b'\x00\x00'
    header.write(RSV)
    STANDALONE = b'\x00'
    header.write(STANDALONE)
    self._write_SOCKS5_address(address, header)
    sent = super(socksocket, self).send(header.getvalue() + bytes, *flags, **kwargs)
    return sent - header.tell()