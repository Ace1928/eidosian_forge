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
def maximum_sctype(t):
    """
    Return the scalar type of highest precision of the same kind as the input.

    Parameters
    ----------
    t : dtype or dtype specifier
        The input data type. This can be a `dtype` object or an object that
        is convertible to a `dtype`.

    Returns
    -------
    out : dtype
        The highest precision data type of the same kind (`dtype.kind`) as `t`.

    See Also
    --------
    obj2sctype, mintypecode, sctype2char
    dtype

    Examples
    --------
    >>> np.maximum_sctype(int)
    <class 'numpy.int64'>
    >>> np.maximum_sctype(np.uint8)
    <class 'numpy.uint64'>
    >>> np.maximum_sctype(complex)
    <class 'numpy.complex256'> # may vary

    >>> np.maximum_sctype(str)
    <class 'numpy.str_'>

    >>> np.maximum_sctype('i2')
    <class 'numpy.int64'>
    >>> np.maximum_sctype('f4')
    <class 'numpy.float128'> # may vary

    """
    g = obj2sctype(t)
    if g is None:
        return t
    t = g
    base = _kind_name(dtype(t))
    if base in sctypes:
        return sctypes[base][-1]
    else:
        return t