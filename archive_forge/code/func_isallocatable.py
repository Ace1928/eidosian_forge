import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def isallocatable(var):
    return 'attrspec' in var and 'allocatable' in var['attrspec']