import os
import sys
from types import ModuleType
from .version import version as __version__  # NOQA:F401
def importobj(modpath, attrname):
    """imports a module, then resolves the attrname on it"""
    module = __import__(modpath, None, None, ['__doc__'])
    if not attrname:
        return module
    retval = module
    names = attrname.split('.')
    for x in names:
        retval = getattr(retval, x)
    return retval