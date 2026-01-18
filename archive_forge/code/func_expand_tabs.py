from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def expand_tabs(line):
    line = line.replace(' ', '\x00')
    line = line.expandtabs(self._tabsize)
    line = line.replace(' ', '\t')
    return line.replace('\x00', ' ').rstrip('\n')