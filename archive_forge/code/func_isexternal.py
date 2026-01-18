import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def isexternal(var):
    return 'attrspec' in var and 'external' in var['attrspec']