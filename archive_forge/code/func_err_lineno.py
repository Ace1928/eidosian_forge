import sys
import re
import unittest
from curtsies.fmtfuncs import bold, green, magenta, cyan, red, plain
from unittest import mock
from bpython.curtsiesfrontend import interpreter
def err_lineno(self, a):
    strings = [x.__unicode__() for x in a]
    for line in reversed(strings):
        clean_line = remove_ansi(line)
        m = re.search('line (\\d+)[,]', clean_line)
        if m:
            return int(m.group(1))
    return None