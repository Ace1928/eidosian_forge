import ast
import sys
import importlib.util
def readmodule_ex(module, path=None):
    """Return a dictionary with all functions and classes in module.

    Search for module in PATH + sys.path.
    If possible, include imported superclasses.
    Do this by reading source, without importing (and executing) it.
    """
    return _readmodule(module, path or [])