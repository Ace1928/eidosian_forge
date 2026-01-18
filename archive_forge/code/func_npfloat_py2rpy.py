import rpy2.robjects as ro
import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
from rpy2.rinterface import (Sexp,
import numpy  # type: ignore
import warnings
@py2rpy.register(numpy.floating)
def npfloat_py2rpy(obj):
    return rinterface.FloatSexpVector([obj])