import warnings
from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.keras.legacy_tf_layers import base
def average_pooling3d(inputs, pool_size, strides, padding='valid', data_format='channels_last', name=None):
    """Average pooling layer for 3D inputs (e.g. volumes).

  Args:
    inputs: The tensor over which to pool. Must have rank 5.
    pool_size: An integer or tuple/list of 3 integers:
      (pool_depth, pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    name: A string, the name of the layer.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
    warnings.warn('`tf.layers.average_pooling3d` is deprecated and will be removed in a future version. Please use `tf.keras.layers.AveragePooling3D` instead.')
    layer = AveragePooling3D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format, name=name)
    return layer.apply(inputs)