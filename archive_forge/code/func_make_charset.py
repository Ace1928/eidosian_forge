import re
from re import escape
from os.path import commonprefix
from itertools import groupby
from operator import itemgetter
def make_charset(letters):
    return '[' + CS_ESCAPE.sub(lambda m: '\\' + m.group(), ''.join(letters)) + ']'