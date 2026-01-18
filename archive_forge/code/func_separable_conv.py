import warnings
import tensorflow as tf
from keras.src.backend import standardize_data_format
from keras.src.backend import standardize_dtype
from keras.src.backend.common.backend_utils import (
from keras.src.backend.config import epsilon
from keras.src.backend.tensorflow.core import cast
from keras.src.backend.tensorflow.core import convert_to_tensor
def separable_conv(inputs, depthwise_kernel, pointwise_kernel, strides=1, padding='valid', data_format=None, dilation_rate=1):
    data_format = standardize_data_format(data_format)
    num_spatial_dims = len(inputs.shape) - 2
    if num_spatial_dims > 2:
        raise ValueError(f'`num_spatial_dims` must be 1 or 2. Received: num_spatial_dims={num_spatial_dims}.')
    tf_data_format = _convert_data_format(data_format, 4)
    padding = padding.upper()
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * num_spatial_dims
    if num_spatial_dims == 1:
        if data_format == 'channels_last':
            strides = (1,) + strides * 2 + (1,)
            spatial_start_dim = 1
        else:
            strides = (1, 1) + strides * 2
            spatial_start_dim = 2
        inputs = tf.expand_dims(inputs, spatial_start_dim)
        depthwise_kernel = tf.expand_dims(depthwise_kernel, axis=0)
        pointwise_kernel = tf.expand_dims(pointwise_kernel, axis=0)
        dilation_rate = None if dilation_rate is None else (1,) + dilation_rate
        outputs = tf.nn.separable_conv2d(inputs, depthwise_kernel, pointwise_kernel, strides, padding, data_format=tf_data_format, dilations=dilation_rate)
        return tf.squeeze(outputs, [spatial_start_dim])
    if data_format == 'channels_last':
        strides = (1,) + strides + (1,)
    else:
        strides = (1, 1) + strides
    return tf.nn.separable_conv2d(inputs, depthwise_kernel, pointwise_kernel, strides, padding, data_format=tf_data_format, dilations=dilation_rate)