import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def issigned_chararray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical'] and (get_kind(var) == '1')