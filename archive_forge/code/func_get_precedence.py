import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def get_precedence(self, node):
    return self._precedences.get(node, _Precedence.TEST)