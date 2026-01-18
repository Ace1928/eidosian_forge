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
def end_struct(self, data):
    mark = self._marks.pop()
    dict = {}
    items = self._stack[mark:]
    for i in range(0, len(items), 2):
        dict[items[i]] = items[i + 1]
    self._stack[mark:] = [dict]
    self._value = 0