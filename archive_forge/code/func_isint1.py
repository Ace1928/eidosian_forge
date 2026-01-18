import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def isint1(var):
    return var.get('typespec') == 'integer' and get_kind(var) == '1' and (not isarray(var))