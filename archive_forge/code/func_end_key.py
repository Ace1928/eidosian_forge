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
def end_key(self):
    if self.current_key or not isinstance(self.stack[-1], type({})):
        raise ValueError('unexpected key at line %d' % self.parser.CurrentLineNumber)
    self.current_key = self.get_data()