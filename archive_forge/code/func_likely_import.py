import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
from tokenize import TokenError
from ._dill import IS_IPYTHON
def likely_import(obj, passive=False, explicit=False):
    return getimport(obj, verify=not passive, builtin=explicit)