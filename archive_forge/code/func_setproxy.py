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
def setproxy(self, *args, **kwargs):
    if 'proxytype' in kwargs:
        kwargs['proxy_type'] = kwargs.pop('proxytype')
    return self.set_proxy(*args, **kwargs)