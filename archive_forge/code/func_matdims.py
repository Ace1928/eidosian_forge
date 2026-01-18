import numpy as np
from scipy._lib import doccer
from . import _byteordercodes as boc
def matdims(arr, oned_as='column'):
    """
    Determine equivalent MATLAB dimensions for given array

    Parameters
    ----------
    arr : ndarray
        Input array
    oned_as : {'column', 'row'}, optional
        Whether 1-D arrays are returned as MATLAB row or column matrices.
        Default is 'column'.

    Returns
    -------
    dims : tuple
        Shape tuple, in the form MATLAB expects it.

    Notes
    -----
    We had to decide what shape a 1 dimensional array would be by
    default. ``np.atleast_2d`` thinks it is a row vector. The
    default for a vector in MATLAB (e.g., ``>> 1:12``) is a row vector.

    Versions of scipy up to and including 0.11 resulted (accidentally)
    in 1-D arrays being read as column vectors. For the moment, we
    maintain the same tradition here.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.io.matlab._miobase import matdims
    >>> matdims(np.array(1)) # NumPy scalar
    (1, 1)
    >>> matdims(np.array([1])) # 1-D array, 1 element
    (1, 1)
    >>> matdims(np.array([1,2])) # 1-D array, 2 elements
    (2, 1)
    >>> matdims(np.array([[2],[3]])) # 2-D array, column vector
    (2, 1)
    >>> matdims(np.array([[2,3]])) # 2-D array, row vector
    (1, 2)
    >>> matdims(np.array([[[2,3]]])) # 3-D array, rowish vector
    (1, 1, 2)
    >>> matdims(np.array([])) # empty 1-D array
    (0, 0)
    >>> matdims(np.array([[]])) # empty 2-D array
    (0, 0)
    >>> matdims(np.array([[[]]])) # empty 3-D array
    (0, 0, 0)

    Optional argument flips 1-D shape behavior.

    >>> matdims(np.array([1,2]), 'row') # 1-D array, 2 elements
    (1, 2)

    The argument has to make sense though

    >>> matdims(np.array([1,2]), 'bizarre')
    Traceback (most recent call last):
       ...
    ValueError: 1-D option "bizarre" is strange

    """
    shape = arr.shape
    if shape == ():
        return (1, 1)
    if len(shape) == 1:
        if shape[0] == 0:
            return (0, 0)
        elif oned_as == 'column':
            return shape + (1,)
        elif oned_as == 'row':
            return (1,) + shape
        else:
            raise ValueError('1-D option "%s" is strange' % oned_as)
    return shape