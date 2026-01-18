import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def markoutercomma(line, comma=','):
    l = ''
    f = 0
    before, after = split_by_unquoted(line, comma + '()')
    l += before
    while after:
        if after[0] == comma and f == 0:
            l += '@' + comma + '@'
        else:
            l += after[0]
            if after[0] == '(':
                f += 1
            elif after[0] == ')':
                f -= 1
        before, after = split_by_unquoted(after[1:], comma + '()')
        l += before
    assert not f, repr((f, line, l))
    return l