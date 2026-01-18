import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def isintent_nothide(var):
    return not isintent_hide(var)