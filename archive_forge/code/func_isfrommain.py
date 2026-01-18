import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
from tokenize import TokenError
from ._dill import IS_IPYTHON
def isfrommain(obj):
    """check if object was built in __main__"""
    module = getmodule(obj)
    if module and module.__name__ == '__main__':
        return True
    return False