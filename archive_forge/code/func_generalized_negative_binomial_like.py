from ._internal import NDArrayBase
from ..base import _Null
def generalized_negative_binomial_like(data=None, mu=_Null, alpha=_Null, out=None, name=None, **kwargs):
    """Draw random samples from a generalized negative binomial distribution according to the
    input array shape.

    Samples are distributed according to a generalized negative binomial distribution parametrized by
    *mu* (mean) and *alpha* (dispersion). *alpha* is defined as *1/k* where *k* is the failure limit of the
    number of unsuccessful experiments (generalized to real numbers).
    Samples will always be returned as a floating point data type.

    Example::

       generalized_negative_binomial(mu=2.0, alpha=0.3, data=ones(2,2)) = [[ 2.,  1.],
                                                                           [ 6.,  4.]]


    Defined in ../src/operator/random/sample_op.cc:L283

    Parameters
    ----------
    mu : float, optional, default=1
        Mean of the negative binomial distribution.
    alpha : float, optional, default=1
        Alpha (dispersion) parameter of the negative binomial distribution.
    data : NDArray
        The input

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)