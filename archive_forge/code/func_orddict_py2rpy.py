import rpy2.robjects as ro
import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
from rpy2.rinterface import (Sexp,
import numpy  # type: ignore
import warnings
@py2rpy.register(rlc.OrdDict)
def orddict_py2rpy(obj):
    rlist = ro.vectors.ListVector.from_length(len(obj))
    rlist.names = ro.vectors.StrVector(tuple(obj.keys()))
    with conversion.get_conversion().context() as cv:
        for i, (k, v) in enumerate(obj.items()):
            rlist[i] = cv.py2rpy(v)
    return rlist