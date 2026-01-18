from ._internal import NDArrayBase
from ..base import _Null
def broadcast_not_equal(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """Returns the result of element-wise **not equal to** (!=) comparison operation with broadcasting.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_not_equal(x, y) = [[ 1.,  1.,  1.],
                                    [ 0.,  0.,  0.]]



    Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L63

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