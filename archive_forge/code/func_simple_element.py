import binascii
import codecs
import datetime
import enum
from io import BytesIO
import itertools
import os
import re
import struct
from xml.parsers.expat import ParserCreate
def simple_element(self, element, value=None):
    if value is not None:
        value = _escape(value)
        self.writeln('<%s>%s</%s>' % (element, value, element))
    else:
        self.writeln('<%s/>' % element)