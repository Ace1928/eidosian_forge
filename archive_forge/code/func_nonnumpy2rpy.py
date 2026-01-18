import rpy2.robjects as ro
import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
from rpy2.rinterface import (Sexp,
import numpy  # type: ignore
import warnings
@py2rpy.register(object)
def nonnumpy2rpy(obj):
    if not isinstance(obj, numpy.ndarray) and hasattr(obj, '__array__'):
        obj = obj.__array__()
        return ro.default_converter.py2rpy(obj)
    elif original_converter is None:
        return ro.default_converter.py2rpy(obj)
    else:
        return original_converter.py2rpy(obj)