import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def isunsigned_char(var):
    if not isscalar(var):
        return 0
    if var.get('typespec') != 'integer':
        return 0
    return get_kind(var) == '-1'