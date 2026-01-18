from .._util import get_backend
from ..util.regression import least_squares
from ..util.pyutil import defaultnamedtuple
from ..units import default_constants, default_units, format_string, patched_numpy
def as_RateExpr(self, unique_keys=None, constants=default_constants, units=default_units):
    return super(ArrheniusParamWithUnits, self).as_RateExpr(unique_keys, constants, units)