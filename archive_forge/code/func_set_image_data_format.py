from tensorflow.python.util import dispatch
def set_image_data_format(data_format):
    """Sets the value of the image data format convention.

  Args:
      data_format: string. `'channels_first'` or `'channels_last'`.

  Example:
  >>> tf.keras.backend.image_data_format()
  'channels_last'
  >>> tf.keras.backend.set_image_data_format('channels_first')
  >>> tf.keras.backend.image_data_format()
  'channels_first'
  >>> tf.keras.backend.set_image_data_format('channels_last')

  Raises:
      ValueError: In case of invalid `data_format` value.
  """
    global _IMAGE_DATA_FORMAT
    if data_format not in {'channels_last', 'channels_first'}:
        raise ValueError('Unknown data_format: ' + str(data_format))
    _IMAGE_DATA_FORMAT = str(data_format)