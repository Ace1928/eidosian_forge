import re, os, stat, io
from xdg.Exceptions import (ParsingError, DuplicateGroupError, NoGroupError,
import xdg.Locale
from xdg.util import u
def checkRegex(self, value):
    try:
        re.compile(value)
    except:
        return 1