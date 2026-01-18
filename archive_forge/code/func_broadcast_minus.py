from ._internal import NDArrayBase
from ..base import _Null
def broadcast_minus(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """Returns element-wise difference of the input arrays with broadcasting.

    `broadcast_minus` is an alias to the function `broadcast_sub`.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_sub(x, y) = [[ 1.,  1.,  1.],
                              [ 0.,  0.,  0.]]

       broadcast_minus(x, y) = [[ 1.,  1.,  1.],
                                [ 0.,  0.,  0.]]

    Supported sparse operations:

       broadcast_sub/minus(csr, dense(1D)) = dense
       broadcast_sub/minus(dense(1D), csr) = dense



    Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L105

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