import re, os, stat, io
from xdg.Exceptions import (ParsingError, DuplicateGroupError, NoGroupError,
import xdg.Locale
from xdg.util import u
def checkPoint(self, value):
    if not re.match('^[0-9]+,[0-9]+$', value):
        return 1