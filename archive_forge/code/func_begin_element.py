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
def begin_element(self, element):
    self.stack.append(element)
    self.writeln('<%s>' % element)
    self._indent_level += 1