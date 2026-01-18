from numpy.core.overrides import ARRAY_FUNCTIONS as _array_functions
from numpy import ufunc as _ufunc
import numpy.core.umath as _umath
def get_overridable_numpy_ufuncs():
    """List all numpy ufuncs overridable via `__array_ufunc__`

    Parameters
    ----------
    None

    Returns
    -------
    set
        A set containing all overridable ufuncs in the public numpy API.
    """
    ufuncs = {obj for obj in _umath.__dict__.values() if isinstance(obj, _ufunc)}
    return ufuncs