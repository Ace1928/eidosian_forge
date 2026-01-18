from ._internal import NDArrayBase
from ..base import _Null
def round_ste(data=None, out=None, name=None, **kwargs):
    """Straight-through-estimator of `round()`.

    In forward pass, returns element-wise rounded value to the nearest integer of the input (same as `round()`).

    In backward pass, returns gradients of ``1`` everywhere (instead of ``0`` everywhere as in `round()`):
    :math:`\\frac{d}{dx}{round\\_ste(x)} = 1` vs. :math:`\\frac{d}{dx}{round(x)} = 0`.
    This is useful for quantized training.

    Reference: Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.

    Example::
      x = round_ste([-1.5, 1.5, -1.9, 1.9, 2.7])
      x.backward()
      x = [-2.,  2., -2.,  2.,  3.]
      x.grad() = [1.,  1., 1.,  1.,  1.]

    The storage type of ``round_ste`` output depends upon the input storage type:
      - round_ste(default) = default
      - round_ste(row_sparse) = row_sparse
      - round_ste(csr) = csr


    Defined in ../src/operator/contrib/stes_op.cc:L54

    Parameters
    ----------
    data : NDArray
        The input array.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)