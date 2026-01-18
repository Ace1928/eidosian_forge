import contextlib
from importlib import import_module
import os
import sys
from . import _util
def preimport(project, modules, **kwargs):
    """Import each of the named modules out of the vendored project."""
    with vendored(project, **kwargs):
        for name in modules:
            import_module(name)