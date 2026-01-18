import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
def set_initial_value(self, y, t=0.0):
    """Set initial conditions y(t) = y."""
    y = asarray(y)
    self.tmp = zeros(y.size * 2, 'float')
    self.tmp[::2] = real(y)
    self.tmp[1::2] = imag(y)
    return ode.set_initial_value(self, self.tmp, t)