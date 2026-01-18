import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def isintent_aligned4(var):
    return 'aligned4' in var.get('intent', [])