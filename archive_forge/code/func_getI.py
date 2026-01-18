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
def getI(self):
    """Matrix inverse or pseudo-inverse."""
    m, n = self.shape
    return (inv if m == n else pinv)(self)