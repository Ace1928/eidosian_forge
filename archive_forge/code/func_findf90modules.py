import numpy as np
from . import capi_maps
from . import func2subr
from .crackfortran import undo_rmbadname, undo_rmbadname1
from .auxfuncs import *
def findf90modules(m):
    if ismodule(m):
        return [m]
    if not hasbody(m):
        return []
    ret = []
    for b in m['body']:
        if ismodule(b):
            ret.append(b)
        else:
            ret = ret + findf90modules(b)
    return ret