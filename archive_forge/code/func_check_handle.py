import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
def check_handle(self):
    if self.handle is not self.__class__.active_global_handle:
        raise IntegratorConcurrencyError(self.__class__.__name__)