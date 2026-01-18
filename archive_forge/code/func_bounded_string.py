import datetime
import io
from os import linesep
import re
import sys
from toml.tz import TomlTz
def bounded_string(self, s):
    if len(s) == 0:
        return True
    if s[-1] != s[0]:
        return False
    i = -2
    backslash = False
    while len(s) + i > 0:
        if s[i] == '\\':
            backslash = not backslash
            i -= 1
        else:
            break
    return not backslash