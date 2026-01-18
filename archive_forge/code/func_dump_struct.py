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
def dump_struct(self, value, write, escape=escape):
    i = id(value)
    if i in self.memo:
        raise TypeError('cannot marshal recursive dictionaries')
    self.memo[i] = None
    dump = self.__dump
    write('<value><struct>\n')
    for k, v in value.items():
        write('<member>\n')
        if not isinstance(k, str):
            raise TypeError('dictionary key must be string')
        write('<name>%s</name>\n' % escape(k))
        dump(v, write)
        write('</member>\n')
    write('</struct></value>\n')
    del self.memo[i]