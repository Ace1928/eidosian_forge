import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def hasinitvalueasstring(var):
    if not hasinitvalue(var):
        return 0
    return var['='][0] in ['"', "'"]