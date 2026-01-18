import warnings
from .._utils import set_module
from ._machar import MachAr
from . import numeric
from . import numerictypes as ntypes
from .numeric import array, inf, NaN
from .umath import log10, exp2, nextafter, isnan
@property
def smallest_subnormal(self):
    """Return the value for the smallest subnormal.

        Returns
        -------
        smallest_subnormal : float
            value for the smallest subnormal.

        Warns
        -----
        UserWarning
            If the calculated value for the smallest subnormal is zero.
        """
    value = self._smallest_subnormal
    if self.ftype(0) == value:
        warnings.warn('The value of the smallest subnormal for {} type is zero.'.format(self.ftype), UserWarning, stacklevel=2)
    return self._float_to_float(value)