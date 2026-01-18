import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def delimit_if(self, start, end, condition):
    if condition:
        return self.delimit(start, end)
    else:
        return nullcontext()