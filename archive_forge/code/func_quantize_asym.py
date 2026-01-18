from ._internal import NDArrayBase
from ..base import _Null
def quantize_asym(data=None, min_calib_range=_Null, max_calib_range=_Null, out=None, name=None, **kwargs):
    """Quantize a input tensor from float to uint8_t.
    Output `scale` and `shift` are scalar floats that specify the quantization parameters for the input
    data.
    The output is calculated using the following equation:
    `out[i] = in[i] * scale + shift + 0.5`,
    where `scale = uint8_range / (max_range - min_range)` and
    `shift = numeric_limits<T>::max - max_range * scale`.
    .. Note::
        This operator only supports forward propagation. DO NOT use it in training.

    Defined in ../src/operator/quantization/quantize_asym.cc:L115

    Parameters
    ----------
    data : NDArray
        A ndarray/symbol of type `float32`
    min_calib_range : float or None, optional, default=None
        The minimum scalar value in the form of float32. If present, it will be used to quantize the fp32 data.
    max_calib_range : float or None, optional, default=None
        The maximum scalar value in the form of float32. If present, it will be used to quantize the fp32 data.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)