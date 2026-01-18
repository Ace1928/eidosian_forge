from ._internal import NDArrayBase
from ..base import _Null
def random_color_jitter(data=None, brightness=_Null, contrast=_Null, saturation=_Null, hue=_Null, out=None, name=None, **kwargs):
    """

    Defined in ../src/operator/image/image_random.cc:L246

    Parameters
    ----------
    data : NDArray
        The input.
    brightness : float, required
        How much to jitter brightness.
    contrast : float, required
        How much to jitter contrast.
    saturation : float, required
        How much to jitter saturation.
    hue : float, required
        How much to jitter hue.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)