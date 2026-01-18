from ._internal import NDArrayBase
from ..base import _Null
def random_pdf_uniform(sample=None, low=None, high=None, is_log=_Null, out=None, name=None, **kwargs):
    """Computes the value of the PDF of *sample* of
    uniform distributions on the intervals given by *[low,high)*.

    *low* and *high* must have the same shape, which must match the leftmost subshape
    of *sample*.  That is, *sample* can have the same shape as *low* and *high*, in which
    case the output contains one density per distribution, or *sample* can be a tensor
    of tensors with that shape, in which case the output is a tensor of densities such that
    the densities at index *i* in the output are given by the samples at index *i* in *sample*
    parameterized by the values of *low* and *high* at index *i*.

    Examples::

        random_pdf_uniform(sample=[[1,2,3,4]], low=[0], high=[10]) = [0.1, 0.1, 0.1, 0.1]

        sample = [[[1, 2, 3],
                   [1, 2, 3]],
                  [[1, 2, 3],
                   [1, 2, 3]]]
        low  = [[0, 0],
                [0, 0]]
        high = [[ 5, 10],
                [15, 20]]
        random_pdf_uniform(sample=sample, low=low, high=high) =
            [[[0.2,        0.2,        0.2    ],
              [0.1,        0.1,        0.1    ]],
             [[0.06667,    0.06667,    0.06667],
              [0.05,       0.05,       0.05   ]]]



    Defined in ../src/operator/random/pdf_op.cc:L297

    Parameters
    ----------
    sample : NDArray
        Samples from the distributions.
    low : NDArray
        Lower bounds of the distributions.
    is_log : boolean, optional, default=0
        If set, compute the density of the log-probability instead of the probability.
    high : NDArray
        Upper bounds of the distributions.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)