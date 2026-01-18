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
def send_content(self, connection, request_body):
    if self.encode_threshold is not None and self.encode_threshold < len(request_body) and gzip:
        connection.putheader('Content-Encoding', 'gzip')
        request_body = gzip_encode(request_body)
    connection.putheader('Content-Length', str(len(request_body)))
    connection.endheaders(request_body)