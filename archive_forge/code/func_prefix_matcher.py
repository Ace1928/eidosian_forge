import contextlib
from importlib import import_module
import os
import sys
from . import _util
def prefix_matcher(*prefixes):
    """Return a module match func that matches any of the given prefixes."""
    assert prefixes

    def match(name, module):
        for prefix in prefixes:
            if name.startswith(prefix):
                return True
        else:
            return False
    return match