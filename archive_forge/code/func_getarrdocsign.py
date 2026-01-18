from . import __version__
import copy
import re
import os
from .crackfortran import markoutercomma
from . import cb_rules
from ._isocbind import iso_c_binding_map, isoc_c2pycode_map, iso_c2py_map
from .auxfuncs import *
def getarrdocsign(a, var):
    ctype = getctype(var)
    if isstring(var) and (not isarray(var)):
        sig = "%s : rank-0 array(string(len=%s),'c')" % (a, getstrlength(var))
    elif isscalar(var):
        sig = "%s : rank-0 array(%s,'%s')" % (a, c2py_map[ctype], c2pycode_map[ctype])
    elif isarray(var):
        dim = var['dimension']
        rank = repr(len(dim))
        sig = "%s : rank-%s array('%s') with bounds (%s)" % (a, rank, c2pycode_map[ctype], ','.join(dim))
    return sig