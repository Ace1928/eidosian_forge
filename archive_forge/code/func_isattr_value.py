import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def isattr_value(var):
    return 'value' in var.get('attrspec', [])