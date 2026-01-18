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
def dump_array(self, value, write):
    i = id(value)
    if i in self.memo:
        raise TypeError('cannot marshal recursive sequences')
    self.memo[i] = None
    dump = self.__dump
    write('<value><array><data>\n')
    for v in value:
        dump(v, write)
    write('</data></array></value>\n')
    del self.memo[i]