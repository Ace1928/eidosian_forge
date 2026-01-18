from ._internal import NDArrayBase
from ..base import _Null
def broadcast_mul(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """Returns element-wise product of the input arrays with broadcasting.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_mul(x, y) = [[ 0.,  0.,  0.],
                              [ 1.,  1.,  1.]]

    Supported sparse operations:

       broadcast_mul(csr, dense(1D)) = csr



    Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L145

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