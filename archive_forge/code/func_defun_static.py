from ..libmp.backend import xrange
import math
import cmath
def defun_static(f):
    setattr(SpecialFunctions, f.__name__, f)
    return f