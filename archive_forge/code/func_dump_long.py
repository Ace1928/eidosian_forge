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
def dump_long(self, value, write):
    if value > MAXINT or value < MININT:
        raise OverflowError('int exceeds XML-RPC limits')
    write('<value><int>')
    write(str(int(value)))
    write('</int></value>\n')