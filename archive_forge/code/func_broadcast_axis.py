from ._internal import NDArrayBase
from ..base import _Null
def broadcast_axis(data=None, axis=_Null, size=_Null, out=None, name=None, **kwargs):
    """Broadcasts the input array over particular axes.

    Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
    `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.

    `broadcast_axes` is an alias to the function `broadcast_axis`.

    Example::

       // given x of shape (1,2,1)
       x = [[[ 1.],
             [ 2.]]]

       // broadcast x on on axis 2
       broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],
                                             [ 2.,  2.,  2.]]]
       // broadcast x on on axes 0 and 2
       broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],
                                                     [ 2.,  2.,  2.]],
                                                    [[ 1.,  1.,  1.],
                                                     [ 2.,  2.,  2.]]]


    Defined in ../src/operator/tensor/broadcast_reduce_op_value.cc:L92

    Parameters
    ----------
    data : NDArray
        The input
    axis : Shape(tuple), optional, default=[]
        The axes to perform the broadcasting.
    size : Shape(tuple), optional, default=[]
        Target sizes of the broadcasting axes.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)