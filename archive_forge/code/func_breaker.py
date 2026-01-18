import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def breaker(instring, loc, doActions=True, callPreParse=True):
    import pdb
    pdb.set_trace()
    return _parseMethod(instring, loc, doActions, callPreParse)