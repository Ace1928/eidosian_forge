from ._internal import NDArrayBase
from ..base import _Null
def quantized_flatten(data=None, min_data=None, max_data=None, out=None, name=None, **kwargs):
    """

    Parameters
    ----------
    data : NDArray
        A ndarray/symbol of type `float32`
    min_data : NDArray
        The minimum scalar value possibly produced for the data
    max_data : NDArray
        The maximum scalar value possibly produced for the data

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)