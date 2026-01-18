from __future__ import division
from builtins import next
from builtins import zip
from builtins import range
import sys
import inspect
import numpy
from numpy.core import numeric
import uncertainties.umath_core as umath_core
import uncertainties.core as uncert_core
from uncertainties.core import deprecation
def array_derivative(array_like, var):
    """
    Return the derivative of the given array with respect to the
    given variable.

    The returned derivative is a NumPy ndarray of the same shape as
    array_like, that contains floats.

    array_like -- array-like object (list, etc.)  that contains
    scalars or numbers with uncertainties.

    var -- Variable object.
    """
    return numpy.vectorize(lambda u: derivative(u, var), otypes=[float])(array_like)