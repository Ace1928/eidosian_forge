import base64
import sys
import time
from datetime import datetime
from decimal import Decimal
import http.client
import urllib.parse
from xml.parsers import expat
import errno
from io import BytesIO
def gzip_decode(data, max_decode=20971520):
    """gzip encoded data -> unencoded data

    Decode data using the gzip content encoding as described in RFC 1952
    """
    if not gzip:
        raise NotImplementedError
    with gzip.GzipFile(mode='rb', fileobj=BytesIO(data)) as gzf:
        try:
            if max_decode < 0:
                decoded = gzf.read()
            else:
                decoded = gzf.read(max_decode + 1)
        except OSError:
            raise ValueError('invalid data')
    if max_decode >= 0 and len(decoded) > max_decode:
        raise ValueError('max gzipped payload length exceeded')
    return decoded