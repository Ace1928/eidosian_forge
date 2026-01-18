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
def end_dict(self):
    if self.current_key:
        raise ValueError("missing value for key '%s' at line %d" % (self.current_key, self.parser.CurrentLineNumber))
    self.stack.pop()