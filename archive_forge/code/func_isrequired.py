import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def isrequired(var):
    return not isoptional(var) and isintent_nothide(var)