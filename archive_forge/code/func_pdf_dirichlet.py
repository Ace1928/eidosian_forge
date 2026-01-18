from ._internal import NDArrayBase
from ..base import _Null
def pdf_dirichlet(sample=None, alpha=None, is_log=_Null, out=None, name=None, **kwargs):
    """Computes the value of the PDF of *sample* of
    Dirichlet distributions with parameter *alpha*.

    The shape of *alpha* must match the leftmost subshape of *sample*.  That is, *sample*
    can have the same shape as *alpha*, in which case the output contains one density per
    distribution, or *sample* can be a tensor of tensors with that shape, in which case
    the output is a tensor of densities such that the densities at index *i* in the output
    are given by the samples at index *i* in *sample* parameterized by the value of *alpha*
    at index *i*.

    Examples::

        random_pdf_dirichlet(sample=[[1,2],[2,3],[3,4]], alpha=[2.5, 2.5]) =
            [38.413498, 199.60245, 564.56085]

        sample = [[[1, 2, 3], [10, 20, 30], [100, 200, 300]],
                  [[0.1, 0.2, 0.3], [0.01, 0.02, 0.03], [0.001, 0.002, 0.003]]]

        random_pdf_dirichlet(sample=sample, alpha=[0.1, 0.4, 0.9]) =
            [[2.3257459e-02, 5.8420084e-04, 1.4674458e-05],
             [9.2589635e-01, 3.6860607e+01, 1.4674468e+03]]


    Defined in ../src/operator/random/pdf_op.cc:L315

    Parameters
    ----------
    sample : NDArray
        Samples from the distributions.
    alpha : NDArray
        Concentration parameters of the distributions.
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