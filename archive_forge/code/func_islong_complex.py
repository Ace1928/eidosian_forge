import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def islong_complex(var):
    if not iscomplex(var):
        return 0
    return get_kind(var) == '32'