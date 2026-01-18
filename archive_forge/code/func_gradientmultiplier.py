from ._internal import NDArrayBase
from ..base import _Null
def gradientmultiplier(data=None, scalar=_Null, is_int=_Null, out=None, name=None, **kwargs):
    """This operator implements the gradient multiplier function.
    In forward pass it acts as an identity transform. During backpropagation it
    multiplies the gradient from the subsequent level by a scalar factor lambda and passes it to
    the preceding layer.


    Defined in ../src/operator/contrib/gradient_multiplier_op.cc:L78

    Parameters
    ----------
    data : NDArray
        The input array.
    scalar : double, optional, default=1
        Scalar input value
    is_int : boolean, optional, default=1
        Indicate whether scalar input is int type

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)