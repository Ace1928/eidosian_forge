import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
@property
def reverse_index(self):
    return len(self) - self.index - 1