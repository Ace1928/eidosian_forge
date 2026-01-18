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
def end_integer(self):
    raw = self.get_data()
    if raw.startswith('0x') or raw.startswith('0X'):
        self.add_object(int(raw, 16))
    else:
        self.add_object(int(raw))