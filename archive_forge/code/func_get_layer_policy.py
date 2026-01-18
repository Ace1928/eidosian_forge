from tensorflow.python.keras.engine import base_layer
def get_layer_policy(layer):
    """Returns the dtype policy of a layer.

  Warning: This function is deprecated. Use
  `tf.keras.layers.Layer.dtype_policy` instead.

  Args:
    layer: A `tf.keras.layers.Layer`.

  Returns:
    The `tf.keras.mixed_precision.Policy` of the layer.
  """
    if not isinstance(layer, base_layer.Layer):
        raise ValueError('get_policy can only be called on a layer, but got: %s' % (layer,))
    return layer.dtype_policy