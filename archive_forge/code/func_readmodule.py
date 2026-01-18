import ast
import sys
import importlib.util
def readmodule(module, path=None):
    """Return Class objects for the top-level classes in module.

    This is the original interface, before Functions were added.
    """
    res = {}
    for key, value in _readmodule(module, path or []).items():
        if isinstance(value, Class):
            res[key] = value
    return res