from ..libmp.backend import xrange
import math
import cmath
def defun_wrapped(f):
    SpecialFunctions.defined_functions[f.__name__] = (f, True)
    return f