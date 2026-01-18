import contextlib
import os
import typing
import rpy2.rinterface as rinterface
import rpy2.rinterface_lib.sexp as sexp
from rpy2.robjects.robject import RObjectMixin
from rpy2.robjects import conversion
@property
def frame(self) -> sexp.SexpEnvironment:
    return conversion.get_conversion().rpy2py(super().frame)