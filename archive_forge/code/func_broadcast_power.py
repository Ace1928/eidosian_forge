from ._internal import NDArrayBase
from ..base import _Null
def broadcast_power(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """Returns result of first array elements raised to powers from second array, element-wise with broadcasting.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_power(x, y) = [[ 2.,  2.,  2.],
                                [ 4.,  4.,  4.]]



    Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L44

    Parameters
    ----------
    lhs : NDArray
        First input to the function
    rhs : NDArray
        Second input to the function

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)