import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def flatlist(lst):
    if isinstance(lst, list):
        return reduce(lambda x, y, f=flatlist: x + f(y), lst, [])
    return [lst]