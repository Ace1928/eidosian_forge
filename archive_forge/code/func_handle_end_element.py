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
def handle_end_element(self, element):
    handler = getattr(self, 'end_' + element, None)
    if handler is not None:
        handler()