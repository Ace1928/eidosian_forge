from tensorflow.python import tf2
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import load as saved_model_load
from tensorflow.python.keras.saving.saved_model import load_context
from tensorflow.python.keras.saving.saved_model import save as saved_model_save
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils.io_utils import path_to_string
def load_model(filepath, custom_objects=None, compile=True, options=None):
    """Loads a model saved via `model.save()`.

  Usage:

  >>> model = tf.keras.Sequential([
  ...     tf.keras.layers.Dense(5, input_shape=(3,)),
  ...     tf.keras.layers.Softmax()])
  >>> model.save('/tmp/model')
  >>> loaded_model = tf.keras.models.load_model('/tmp/model')
  >>> x = tf.random.uniform((10, 3))
  >>> assert np.allclose(model.predict(x), loaded_model.predict(x))

  Note that the model weights may have different scoped names after being
  loaded. Scoped names include the model/layer names, such as
  `"dense_1/kernel:0"`. It is recommended that you use the layer properties to
  access specific variables, e.g. `model.get_layer("dense_1").kernel`.

  Args:
      filepath: One of the following:
          - String or `pathlib.Path` object, path to the saved model
          - `h5py.File` object from which to load the model
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.
      compile: Boolean, whether to compile the model
          after loading.
      options: Optional `tf.saved_model.LoadOptions` object that specifies
        options for loading from SavedModel.

  Returns:
      A Keras model instance. If the original model was compiled, and saved with
      the optimizer, then the returned model will be compiled. Otherwise, the
      model will be left uncompiled. In the case that an uncompiled model is
      returned, a warning is displayed if the `compile` argument is set to
      `True`.

  Raises:
      ImportError: if loading from an hdf5 file and h5py is not available.
      IOError: In case of an invalid savefile.
  """
    with generic_utils.SharedObjectLoadingScope():
        with generic_utils.CustomObjectScope(custom_objects or {}):
            with load_context.load_context(options):
                if h5py is not None and (isinstance(filepath, h5py.File) or h5py.is_hdf5(filepath)):
                    return hdf5_format.load_model_from_hdf5(filepath, custom_objects, compile)
                filepath = path_to_string(filepath)
                if isinstance(filepath, str):
                    return saved_model_load.load(filepath, compile, options)
    raise IOError('Unable to load model. Filepath is not an hdf5 file (or h5py is not available) or SavedModel.')