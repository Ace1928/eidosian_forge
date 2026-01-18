from ._internal import NDArrayBase
from ..base import _Null
def broadcast_plus(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """Returns element-wise sum of the input arrays with broadcasting.

    `broadcast_plus` is an alias to the function `broadcast_add`.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_add(x, y) = [[ 1.,  1.,  1.],
                              [ 2.,  2.,  2.]]

       broadcast_plus(x, y) = [[ 1.,  1.,  1.],
                               [ 2.,  2.,  2.]]

    Supported sparse operations:

       broadcast_add(csr, dense(1D)) = dense
       broadcast_add(dense(1D), csr) = dense



    Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L57

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