from ._internal import NDArrayBase
from ..base import _Null
def random_pdf_poisson(sample=None, lam=None, is_log=_Null, out=None, name=None, **kwargs):
    """Computes the value of the PDF of *sample* of
    Poisson distributions with parameters *lam* (rate).

    The shape of *lam* must match the leftmost subshape of *sample*.  That is, *sample*
    can have the same shape as *lam*, in which case the output contains one density per
    distribution, or *sample* can be a tensor of tensors with that shape, in which case
    the output is a tensor of densities such that the densities at index *i* in the output
    are given by the samples at index *i* in *sample* parameterized by the value of *lam*
    at index *i*.

    Examples::

        random_pdf_poisson(sample=[[0,1,2,3]], lam=[1]) =
            [[0.36787945, 0.36787945, 0.18393973, 0.06131324]]

        sample = [[0,1,2,3],
                  [0,1,2,3],
                  [0,1,2,3]]

        random_pdf_poisson(sample=sample, lam=[1,2,3]) =
            [[0.36787945, 0.36787945, 0.18393973, 0.06131324],
             [0.13533528, 0.27067056, 0.27067056, 0.18044704],
             [0.04978707, 0.14936121, 0.22404182, 0.22404182]]


    Defined in ../src/operator/random/pdf_op.cc:L306

    Parameters
    ----------
    sample : NDArray
        Samples from the distributions.
    lam : NDArray
        Lambda (rate) parameters of the distributions.
    is_log : boolean, optional, default=0
        If set, compute the density of the log-probability instead of the probability.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)