import re, os, stat, io
from xdg.Exceptions import (ParsingError, DuplicateGroupError, NoGroupError,
import xdg.Locale
from xdg.util import u
def checkString(self, value):
    return 0 if is_ascii(value) else 1