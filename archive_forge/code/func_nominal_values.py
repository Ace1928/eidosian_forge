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
@property
def nominal_values(self):
    """
        Nominal value of all the elements of the matrix.
        """
    return nominal_values(self)