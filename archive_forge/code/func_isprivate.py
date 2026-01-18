import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def isprivate(var):
    return 'attrspec' in var and 'private' in var['attrspec']