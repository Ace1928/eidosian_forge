import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def containscommon(rout):
    if hascommon(rout):
        return 1
    if hasbody(rout):
        for b in rout['body']:
            if containscommon(b):
                return 1
    return 0