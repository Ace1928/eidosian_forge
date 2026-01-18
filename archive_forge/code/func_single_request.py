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
def single_request(self, host, handler, request_body, verbose=False):
    try:
        http_conn = self.send_request(host, handler, request_body, verbose)
        resp = http_conn.getresponse()
        if resp.status == 200:
            self.verbose = verbose
            return self.parse_response(resp)
    except Fault:
        raise
    except Exception:
        self.close()
        raise
    if resp.getheader('content-length', ''):
        resp.read()
    raise ProtocolError(host + handler, resp.status, resp.reason, dict(resp.getheaders()))