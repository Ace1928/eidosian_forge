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
def end_boolean(self, data):
    if data == '0':
        self.append(False)
    elif data == '1':
        self.append(True)
    else:
        raise TypeError('bad boolean value')
    self._value = 0