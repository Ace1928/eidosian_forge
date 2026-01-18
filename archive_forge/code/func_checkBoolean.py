import re, os, stat, io
from xdg.Exceptions import (ParsingError, DuplicateGroupError, NoGroupError,
import xdg.Locale
from xdg.util import u
def checkBoolean(self, value):
    if value == '1' or value == '0':
        return 2
    elif not (value == 'true' or value == 'false'):
        return 1