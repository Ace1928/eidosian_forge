import __future__
from ast import PyCF_ONLY_AST
import codeop
import functools
import hashlib
import linecache
import operator
import time
from contextlib import contextmanager
@property
def compiler_flags(self):
    """Flags currently active in the compilation process.
        """
    return self.flags