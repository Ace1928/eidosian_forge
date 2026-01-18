import math
from .._util import get_backend
from ..util.pyutil import defaultnamedtuple
from ..units import default_units, Backend, default_constants, format_string
from .arrhenius import _get_R, _fit
def dH_over_R(self, constants=None, units=None, backend=None):
    R = _get_R(constants, units)
    return self.dH / R