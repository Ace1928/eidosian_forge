import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def islong_double(var):
    if not isscalar(var):
        return 0
    if not var.get('typespec') == 'real':
        return 0
    return get_kind(var) == '16'