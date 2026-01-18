import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def getusercode1(rout):
    return getmultilineblock(rout, 'usercode', counter=1)