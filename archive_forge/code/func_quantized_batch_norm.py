from ._internal import NDArrayBase
from ..base import _Null
def quantized_batch_norm(data=None, gamma=None, beta=None, moving_mean=None, moving_var=None, min_data=None, max_data=None, eps=_Null, momentum=_Null, fix_gamma=_Null, use_global_stats=_Null, output_mean_var=_Null, axis=_Null, cudnn_off=_Null, min_calib_range=_Null, max_calib_range=_Null, out=None, name=None, **kwargs):
    """BatchNorm operator for input and output data type of int8.
    The input and output data comes with min and max thresholds for quantizing
    the float32 data into int8.

    .. Note::
        This operator only supports forward propogation. DO NOT use it in training.


    Defined in ../src/operator/quantization/quantized_batch_norm.cc:L94

    Parameters
    ----------
    data : NDArray
        Input data.
    gamma : NDArray
        gamma.
    beta : NDArray
        beta.
    moving_mean : NDArray
        moving_mean.
    moving_var : NDArray
        moving_var.
    min_data : NDArray
        Minimum value of data.
    max_data : NDArray
        Maximum value of data.
    eps : double, optional, default=0.0010000000474974513
        Epsilon to prevent div 0. Must be no less than CUDNN_BN_MIN_EPSILON defined in cudnn.h when using cudnn (usually 1e-5)
    momentum : float, optional, default=0.899999976
        Momentum for moving average
    fix_gamma : boolean, optional, default=1
        Fix gamma while training
    use_global_stats : boolean, optional, default=0
        Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.
    output_mean_var : boolean, optional, default=0
        Output the mean and inverse std 
    axis : int, optional, default='1'
        Specify which shape axis the channel is specified
    cudnn_off : boolean, optional, default=0
        Do not select CUDNN operator, if available
    min_calib_range : float or None, optional, default=None
        The minimum scalar value in the form of float32 obtained through calibration. If present, it will be used to by quantized batch norm op to calculate primitive scale.Note: this calib_range is to calib bn output.
    max_calib_range : float or None, optional, default=None
        The maximum scalar value in the form of float32 obtained through calibration. If present, it will be used to by quantized batch norm op to calculate primitive scale.Note: this calib_range is to calib bn output.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)