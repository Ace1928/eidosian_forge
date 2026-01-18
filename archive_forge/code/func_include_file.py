import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def include_file(self, uri, **kwargs):
    """Include a file at the given ``uri``."""
    _include_file(self.context, uri, self._templateuri, **kwargs)