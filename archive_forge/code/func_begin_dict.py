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
def begin_dict(self, attrs):
    d = self._dict_type()
    self.add_object(d)
    self.stack.append(d)