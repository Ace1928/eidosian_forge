from ._internal import NDArrayBase
from ..base import _Null
def broadcast_logical_or(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """Returns the result of element-wise **logical or** with broadcasting.

    Example::

       x = [[ 1.,  1.,  0.],
            [ 1.,  1.,  0.]]

       y = [[ 1.],
            [ 0.]]

       broadcast_logical_or(x, y) = [[ 1.,  1.,  1.],
                                     [ 1.,  1.,  0.]]



    Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L171

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