from ._internal import NDArrayBase
from ..base import _Null
def smooth_l1(data=None, scalar=_Null, out=None, name=None, **kwargs):
    """Calculate Smooth L1 Loss(lhs, scalar) by summing

    .. math::

        f(x) =
        \\begin{cases}
        (\\sigma x)^2/2,& \\text{if }x < 1/\\sigma^2\\\\
        |x|-0.5/\\sigma^2,& \\text{otherwise}
        \\end{cases}

    where :math:`x` is an element of the tensor *lhs* and :math:`\\sigma` is the scalar.

    Example::

      smooth_l1([1, 2, 3, 4]) = [0.5, 1.5, 2.5, 3.5]
      smooth_l1([1, 2, 3, 4], scalar=1) = [0.5, 1.5, 2.5, 3.5]



    Defined in ../src/operator/tensor/elemwise_binary_scalar_op_extended.cc:L108

    Parameters
    ----------
    data : NDArray
        source input
    scalar : float
        scalar input

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)