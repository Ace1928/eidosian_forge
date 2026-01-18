import __future__
from ast import PyCF_ONLY_AST
import codeop
import functools
import hashlib
import linecache
import operator
import time
from contextlib import contextmanager
def reset_compiler_flags(self):
    """Reset compiler flags to default state."""
    self.flags = codeop.PyCF_DONT_IMPLY_DEDENT