import re, os, stat, io
from xdg.Exceptions import (ParsingError, DuplicateGroupError, NoGroupError,
import xdg.Locale
from xdg.util import u
def checkNumber(self, value):
    try:
        float(value)
    except:
        return 1