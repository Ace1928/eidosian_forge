import rpy2.robjects as ro
import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
from rpy2.rinterface import (Sexp,
import numpy  # type: ignore
import warnings
@rpy2py.register(Sexp)
def rpy2py_sexp(obj):
    if obj.typeof in _vectortypes and obj.typeof != RTYPES.VECSXP:
        res = numpy.array(obj)
    else:
        res = ro.default_converter.rpy2py(obj)
    return res