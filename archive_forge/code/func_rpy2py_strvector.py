import rpy2.robjects as ro
import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
from rpy2.rinterface import (Sexp,
import numpy  # type: ignore
import warnings
@rpy2py.register(rinterface.StrSexpVector)
def rpy2py_strvector(obj):
    res = numpy.array(obj)
    res[res == rinterface.NA_Character] = None
    return res