from copy import deepcopy
from abc import ABC, abstractmethod
from types import ModuleType
from typing import (
import sys
import token, tokenize
import os
from os import path
from collections import defaultdict
from functools import partial
from argparse import ArgumentParser
import lark
from lark.tools import lalr_argparser, build_lalr, make_warnings_comments
from lark.grammar import Rule
from lark.lexer import TerminalDef
def strip_docstrings(line_gen):
    """ Strip comments and docstrings from a file.
    Based on code from: https://stackoverflow.com/questions/1769332/script-to-remove-python-comments-docstrings
    """
    res = []
    prev_toktype = token.INDENT
    last_lineno = -1
    last_col = 0
    tokgen = tokenize.generate_tokens(line_gen)
    for toktype, ttext, (slineno, scol), (elineno, ecol), ltext in tokgen:
        if slineno > last_lineno:
            last_col = 0
        if scol > last_col:
            res.append(' ' * (scol - last_col))
        if toktype == token.STRING and prev_toktype == token.INDENT:
            res.append('#--')
        elif toktype == tokenize.COMMENT:
            res.append('##\n')
        else:
            res.append(ttext)
        prev_toktype = toktype
        last_col = ecol
        last_lineno = elineno
    return ''.join(res)