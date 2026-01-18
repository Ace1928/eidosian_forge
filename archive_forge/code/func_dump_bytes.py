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
def dump_bytes(self, value, write):
    write('<value><base64>\n')
    encoded = base64.encodebytes(value)
    write(encoded.decode('ascii'))
    write('</base64></value>\n')