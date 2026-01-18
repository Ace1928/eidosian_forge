import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def set_section(in_section, this_section):
    for entry in in_section.scalars:
        this_section[entry] = in_section[entry]
    for section in in_section.sections:
        this_section[section] = {}
        set_section(in_section[section], this_section[section])