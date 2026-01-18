import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def getrestdoc(rout):
    if 'f2pymultilines' not in rout:
        return None
    k = None
    if rout['block'] == 'python module':
        k = (rout['block'], rout['name'])
    return rout['f2pymultilines'].get(k, None)