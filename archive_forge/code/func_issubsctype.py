import numbers
import warnings
from .multiarray import (
from .._utils import set_module
from ._string_helpers import (
from ._type_aliases import (
from ._dtype import _kind_name
from builtins import bool, int, float, complex, object, str, bytes
from numpy.compat import long, unicode
@set_module('numpy')
def issubsctype(arg1, arg2):
    """
    Determine if the first argument is a subclass of the second argument.

    Parameters
    ----------
    arg1, arg2 : dtype or dtype specifier
        Data-types.

    Returns
    -------
    out : bool
        The result.

    See Also
    --------
    issctype, issubdtype, obj2sctype

    Examples
    --------
    >>> np.issubsctype('S8', str)
    False
    >>> np.issubsctype(np.array([1]), int)
    True
    >>> np.issubsctype(np.array([1]), float)
    False

    """
    return issubclass(obj2sctype(arg1), obj2sctype(arg2))