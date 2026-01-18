from ._internal import NDArrayBase
from ..base import _Null
def broadcast_mod(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """Returns element-wise modulo of the input arrays with broadcasting.

    Example::

       x = [[ 8.,  8.,  8.],
            [ 8.,  8.,  8.]]

       y = [[ 2.],
            [ 3.]]

       broadcast_mod(x, y) = [[ 0.,  0.,  0.],
                              [ 2.,  2.,  2.]]



    Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L221

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