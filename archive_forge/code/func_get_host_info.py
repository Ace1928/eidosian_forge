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
def get_host_info(self, host):
    x509 = {}
    if isinstance(host, tuple):
        host, x509 = host
    auth, host = urllib.parse._splituser(host)
    if auth:
        auth = urllib.parse.unquote_to_bytes(auth)
        auth = base64.encodebytes(auth).decode('utf-8')
        auth = ''.join(auth.split())
        extra_headers = [('Authorization', 'Basic ' + auth)]
    else:
        extra_headers = []
    return (host, extra_headers, x509)