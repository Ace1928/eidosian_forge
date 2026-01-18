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
def dump_datetime(self, value, write):
    write('<value><dateTime.iso8601>')
    write(_strftime(value))
    write('</dateTime.iso8601></value>\n')