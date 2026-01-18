import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def isarrayofstrings(var):
    return isstringarray(var) and var['dimension'][-1] == '(*)'