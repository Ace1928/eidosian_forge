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
def end_dispatch(self, tag, data):
    try:
        f = self.dispatch[tag]
    except KeyError:
        if ':' not in tag:
            return
        try:
            f = self.dispatch[tag.split(':')[-1]]
        except KeyError:
            return
    return f(self, data)