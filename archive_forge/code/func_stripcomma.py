import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def stripcomma(s):
    if s and s[-1] == ',':
        return s[:-1]
    return s