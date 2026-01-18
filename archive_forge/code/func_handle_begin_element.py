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
def handle_begin_element(self, element, attrs):
    self.data = []
    handler = getattr(self, 'begin_' + element, None)
    if handler is not None:
        handler(attrs)