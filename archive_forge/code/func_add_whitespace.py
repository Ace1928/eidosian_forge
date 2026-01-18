from builtins import open as _builtin_open
from codecs import lookup, BOM_UTF8
import collections
import functools
from io import TextIOWrapper
import itertools as _itertools
import re
import sys
from token import *
from token import EXACT_TOKEN_TYPES
import token
def add_whitespace(self, start):
    row, col = start
    if row < self.prev_row or (row == self.prev_row and col < self.prev_col):
        raise ValueError('start ({},{}) precedes previous end ({},{})'.format(row, col, self.prev_row, self.prev_col))
    row_offset = row - self.prev_row
    if row_offset:
        self.tokens.append('\\\n' * row_offset)
        self.prev_col = 0
    col_offset = col - self.prev_col
    if col_offset:
        self.tokens.append(' ' * col_offset)