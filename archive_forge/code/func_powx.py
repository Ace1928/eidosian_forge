from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def powx(x, p):
    return x ** p