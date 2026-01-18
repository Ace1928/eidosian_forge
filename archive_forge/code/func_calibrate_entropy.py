from ._internal import NDArrayBase
from ..base import _Null
def calibrate_entropy(hist=None, hist_edges=None, num_quantized_bins=_Null, out=None, name=None, **kwargs):
    """Provide calibrated min/max for input histogram.

    .. Note::
        This operator only supports forward propagation. DO NOT use it in training.

    Defined in ../src/operator/quantization/calibrate.cc:L196

    Parameters
    ----------
    hist : NDArray
        A ndarray/symbol of type `float32`
    hist_edges : NDArray
        A ndarray/symbol of type `float32`
    num_quantized_bins : int, optional, default='255'
        The number of quantized bins.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)