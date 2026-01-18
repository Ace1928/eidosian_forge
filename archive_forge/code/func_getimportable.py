import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
from tokenize import TokenError
from ._dill import IS_IPYTHON
def getimportable(obj, alias='', byname=True, explicit=False):
    return importable(obj, alias, source=not byname, builtin=explicit)