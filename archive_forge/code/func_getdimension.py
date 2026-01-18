import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def getdimension(var):
    dimpattern = '\\((.*?)\\)'
    if 'attrspec' in var.keys():
        if any(('dimension' in s for s in var['attrspec'])):
            return [re.findall(dimpattern, v) for v in var['attrspec']][0]