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
def obj2sctype(rep, default=None):
    """
    Return the scalar dtype or NumPy equivalent of Python type of an object.

    Parameters
    ----------
    rep : any
        The object of which the type is returned.
    default : any, optional
        If given, this is returned for objects whose types can not be
        determined. If not given, None is returned for those objects.

    Returns
    -------
    dtype : dtype or Python type
        The data type of `rep`.

    See Also
    --------
    sctype2char, issctype, issubsctype, issubdtype, maximum_sctype

    Examples
    --------
    >>> np.obj2sctype(np.int32)
    <class 'numpy.int32'>
    >>> np.obj2sctype(np.array([1., 2.]))
    <class 'numpy.float64'>
    >>> np.obj2sctype(np.array([1.j]))
    <class 'numpy.complex128'>

    >>> np.obj2sctype(dict)
    <class 'numpy.object_'>
    >>> np.obj2sctype('string')

    >>> np.obj2sctype(1, default=list)
    <class 'list'>

    """
    if isinstance(rep, type) and issubclass(rep, generic):
        return rep
    if isinstance(rep, ndarray):
        return rep.dtype.type
    try:
        res = dtype(rep)
    except Exception:
        return default
    else:
        return res.type