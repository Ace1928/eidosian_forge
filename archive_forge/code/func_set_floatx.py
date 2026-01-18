from tensorflow.python.util import dispatch
def set_floatx(value):
    """Sets the default float type.

  Note: It is not recommended to set this to float16 for training, as this will
  likely cause numeric stability issues. Instead, mixed precision, which is
  using a mix of float16 and float32, can be used by calling
  `tf.keras.mixed_precision.set_global_policy('mixed_float16')`. See the
  [mixed precision guide](
    https://www.tensorflow.org/guide/keras/mixed_precision) for details.

  Args:
      value: String; `'float16'`, `'float32'`, or `'float64'`.

  Example:
  >>> tf.keras.backend.floatx()
  'float32'
  >>> tf.keras.backend.set_floatx('float64')
  >>> tf.keras.backend.floatx()
  'float64'
  >>> tf.keras.backend.set_floatx('float32')

  Raises:
      ValueError: In case of invalid value.
  """
    global _FLOATX
    if value not in {'float16', 'float32', 'float64'}:
        raise ValueError('Unknown floatx type: ' + str(value))
    _FLOATX = str(value)