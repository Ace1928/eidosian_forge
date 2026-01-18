import numpy as _np
import functools
from scipy.linalg import _fblas
from scipy.linalg._fblas import *  # noqa: E402, F403
@_memoize_get_funcs
def get_blas_funcs(names, arrays=(), dtype=None, ilp64=False):
    """Return available BLAS function objects from names.

    Arrays are used to determine the optimal prefix of BLAS routines.

    Parameters
    ----------
    names : str or sequence of str
        Name(s) of BLAS functions without type prefix.

    arrays : sequence of ndarrays, optional
        Arrays can be given to determine optimal prefix of BLAS
        routines. If not given, double-precision routines will be
        used, otherwise the most generic type in arrays will be used.

    dtype : str or dtype, optional
        Data-type specifier. Not used if `arrays` is non-empty.

    ilp64 : {True, False, 'preferred'}, optional
        Whether to return ILP64 routine variant.
        Choosing 'preferred' returns ILP64 routine if available,
        and otherwise the 32-bit routine. Default: False

    Returns
    -------
    funcs : list
        List containing the found function(s).


    Notes
    -----
    This routine automatically chooses between Fortran/C
    interfaces. Fortran code is used whenever possible for arrays with
    column major order. In all other cases, C code is preferred.

    In BLAS, the naming convention is that all functions start with a
    type prefix, which depends on the type of the principal
    matrix. These can be one of {'s', 'd', 'c', 'z'} for the NumPy
    types {float32, float64, complex64, complex128} respectively.
    The code and the dtype are stored in attributes `typecode` and `dtype`
    of the returned functions.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.linalg as LA
    >>> rng = np.random.default_rng()
    >>> a = rng.random((3,2))
    >>> x_gemv = LA.get_blas_funcs('gemv', (a,))
    >>> x_gemv.typecode
    'd'
    >>> x_gemv = LA.get_blas_funcs('gemv',(a*1j,))
    >>> x_gemv.typecode
    'z'

    """
    if isinstance(ilp64, str):
        if ilp64 == 'preferred':
            ilp64 = HAS_ILP64
        else:
            raise ValueError("Invalid value for 'ilp64'")
    if not ilp64:
        return _get_funcs(names, arrays, dtype, 'BLAS', _fblas, _cblas, 'fblas', 'cblas', _blas_alias, ilp64=False)
    else:
        if not HAS_ILP64:
            raise RuntimeError('BLAS ILP64 routine requested, but Scipy compiled only with 32-bit BLAS')
        return _get_funcs(names, arrays, dtype, 'BLAS', _fblas_64, None, 'fblas_64', None, _blas_alias, ilp64=True)