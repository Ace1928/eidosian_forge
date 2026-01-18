import contextlib
from importlib import import_module
import os
import sys
from . import _util
def check_modules(project, match, root=None):
    """Verify that only vendored modules have been imported."""
    if root is None:
        root = project_root(project)
    extensions = []
    unvendored = {}
    for modname, mod in list(sys.modules.items()):
        if not match(modname, mod):
            continue
        try:
            filename = getattr(mod, '__file__', None)
        except:
            filename = None
        if not filename:
            extensions.append(modname)
        elif not filename.startswith(root):
            unvendored[modname] = filename
    return (unvendored, extensions)