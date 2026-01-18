import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def increasing_level_traverse(node):
    nonlocal operator_precedence
    operator_precedence = operator_precedence.next()
    self.set_precedence(operator_precedence, node)
    self.traverse(node)