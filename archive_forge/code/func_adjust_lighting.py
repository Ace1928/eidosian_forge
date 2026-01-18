from ._internal import NDArrayBase
from ..base import _Null
def adjust_lighting(data=None, alpha=_Null, out=None, name=None, **kwargs):
    """Adjust the lighting level of the input. Follow the AlexNet style.

    Defined in ../src/operator/image/image_random.cc:L254

    Parameters
    ----------
    data : NDArray
        The input.
    alpha : tuple of <float>, required
        The lighting alphas for the R, G, B channels.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)