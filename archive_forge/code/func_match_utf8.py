import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def match_utf8(encoding):
    return BOM_LIST.get(encoding.lower()) == 'utf_8'