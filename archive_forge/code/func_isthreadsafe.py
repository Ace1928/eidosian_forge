import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def isthreadsafe(rout):
    return 'f2pyenhancements' in rout and 'threadsafe' in rout['f2pyenhancements']