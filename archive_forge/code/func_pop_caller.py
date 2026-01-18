import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def pop_caller(self):
    """Pop a ``caller`` callable onto the callstack for this
        :class:`.Context`."""
    del self.caller_stack[-1]