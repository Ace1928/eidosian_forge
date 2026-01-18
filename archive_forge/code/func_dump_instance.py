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
def dump_instance(self, value, write):
    if value.__class__ in WRAPPERS:
        self.write = write
        value.encode(self)
        del self.write
    else:
        self.dump_struct(value.__dict__, write)