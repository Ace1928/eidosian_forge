from ._internal import NDArrayBase
from ..base import _Null
def quantized_elemwise_mul(lhs=None, rhs=None, lhs_min=None, lhs_max=None, rhs_min=None, rhs_max=None, min_calib_range=_Null, max_calib_range=_Null, enable_float_output=_Null, out=None, name=None, **kwargs):
    """Multiplies arguments int8 element-wise.


    Defined in ../src/operator/quantization/quantized_elemwise_mul.cc:L221

    Parameters
    ----------
    lhs : NDArray
        first input
    rhs : NDArray
        second input
    lhs_min : NDArray
        Minimum value of first input.
    lhs_max : NDArray
        Maximum value of first input.
    rhs_min : NDArray
        Minimum value of second input.
    rhs_max : NDArray
        Maximum value of second input.
    min_calib_range : float or None, optional, default=None
        The minimum scalar value in the form of float32 obtained through calibration. If present, it will be used to requantize the int8 output data.
    max_calib_range : float or None, optional, default=None
        The maximum scalar value in the form of float32 obtained through calibration. If present, it will be used to requantize the int8 output data.
    enable_float_output : boolean, optional, default=0
        Whether to enable float32 output

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)