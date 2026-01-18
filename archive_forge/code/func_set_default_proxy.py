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
def set_default_proxy(proxy_type=None, addr=None, port=None, rdns=True, username=None, password=None):
    """Sets a default proxy.

    All further socksocket objects will use the default unless explicitly
    changed. All parameters are as for socket.set_proxy()."""
    if hasattr(username, 'encode'):
        username = username.encode()
    if hasattr(password, 'encode'):
        password = password.encode()
    socksocket.default_proxy = (proxy_type, addr, port, rdns, username if username else None, password if password else None)