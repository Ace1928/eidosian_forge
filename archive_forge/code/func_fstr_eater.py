from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import contextlib
import itertools
import tokenize
from six import StringIO
from pasta.base import formatting as fmt
from pasta.base import fstring_utils
def fstr_eater(tok):
    if tok.type == TOKENS.OP and tok.src == '}':
        if fstr_eater.level <= 0:
            return False
        fstr_eater.level -= 1
    if tok.type == TOKENS.OP and tok.src == '{':
        fstr_eater.level += 1
    return True