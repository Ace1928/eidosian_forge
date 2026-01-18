from ._internal import NDArrayBase
from ..base import _Null
def multi_all_finite(*data, **kwargs):
    """Check if all the float numbers in all the arrays are finite (used for AMP)


    Defined in ../src/operator/contrib/all_finite.cc:L132

    Parameters
    ----------
    data : NDArray[]
        Arrays
    num_arrays : int, optional, default='1'
        Number of arrays.
    init_output : boolean, optional, default=1
        Initialize output to 1.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)