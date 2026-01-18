from __future__ import division
import uncertainties
import uncertainties.core as uncert_core
from uncertainties import ufloat, unumpy, test_uncertainties
from uncertainties.unumpy import core
from uncertainties.test_uncertainties import numbers_close, arrays_close
def derivatives_close(x, y):
    """
    Returns True iff the AffineScalarFunc objects x and y have
    derivatives that are close to each other (they must depend
    on the same variables).
    """
    if set(x.derivatives) != set(y.derivatives):
        return False
    return all((numbers_close(x.derivatives[var], y.derivatives[var]) for var in x.derivatives))