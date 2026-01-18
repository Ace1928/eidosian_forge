from ._internal import NDArrayBase
from ..base import _Null
def random_lighting(data=None, alpha_std=_Null, out=None, name=None, **kwargs):
    """Randomly add PCA noise. Follow the AlexNet style.

    Defined in ../src/operator/image/image_random.cc:L262

    Parameters
    ----------
    data : NDArray
        The input.
    alpha_std : float, optional, default=0.0500000007
        Level of the lighting noise.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)