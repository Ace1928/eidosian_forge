from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
def ambiguous_identifier(logical_line, tokens):
    """Never use the characters 'l', 'O', or 'I' as variable names.

    In some fonts, these characters are indistinguishable from the numerals
    one and zero. When tempted to use 'l', use 'L' instead.

    Okay: L = 0
    Okay: o = 123
    Okay: i = 42
    E741: l = 0
    E741: O = 123
    E741: I = 42

    Variables can be bound in several other contexts, including class and
    function definitions, 'global' and 'nonlocal' statements, exception
    handlers, and 'with' statements.

    Okay: except AttributeError as o:
    Okay: with lock as L:
    E741: except AttributeError as O:
    E741: with lock as l:
    E741: global I
    E741: nonlocal l
    E742: class I(object):
    E743: def l(x):
    """
    idents_to_avoid = ('l', 'O', 'I')
    prev_type, prev_text, prev_start, prev_end, __ = tokens[0]
    for token_type, text, start, end, line in tokens[1:]:
        ident = pos = None
        if token_type == tokenize.OP and '=' in text:
            if prev_text in idents_to_avoid:
                ident = prev_text
                pos = prev_start
        if prev_text in ('as', 'global', 'nonlocal'):
            if text in idents_to_avoid:
                ident = text
                pos = start
        if prev_text == 'class':
            if text in idents_to_avoid:
                yield (start, "E742 ambiguous class definition '%s'" % text)
        if prev_text == 'def':
            if text in idents_to_avoid:
                yield (start, "E743 ambiguous function definition '%s'" % text)
        if ident:
            yield (pos, "E741 ambiguous variable name '%s'" % ident)
        prev_text = text
        prev_start = start