import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def require_parens(self, precedence, node):
    """Shortcut to adding precedence related parens"""
    return self.delimit_if('(', ')', self.get_precedence(node) > precedence)