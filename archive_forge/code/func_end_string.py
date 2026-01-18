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
def end_string(self, data):
    if self._encoding:
        data = data.decode(self._encoding)
    self.append(data)
    self._value = 0