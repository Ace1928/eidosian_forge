from tensorflow.python.keras.saving.saved_model import json_utils
def model_from_json(json_string, custom_objects=None):
    """Parses a JSON model configuration string and returns a model instance.

  Usage:

  >>> model = tf.keras.Sequential([
  ...     tf.keras.layers.Dense(5, input_shape=(3,)),
  ...     tf.keras.layers.Softmax()])
  >>> config = model.to_json()
  >>> loaded_model = tf.keras.models.model_from_json(config)

  Args:
      json_string: JSON string encoding a model configuration.
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.

  Returns:
      A Keras model instance (uncompiled).
  """
    config = json_utils.decode(json_string)
    from tensorflow.python.keras.layers import deserialize
    return deserialize(config, custom_objects=custom_objects)