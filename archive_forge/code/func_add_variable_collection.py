import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src.engine import base_layer
from keras.src.engine import functional
from keras.src.engine import sequential
from keras.src.utils import io_utils
def add_variable_collection(self, name, variables):
    """Register a set of variables to be retrieved after reloading.

        Arguments:
            name: The string name for the collection.
            variables: A tuple/list/set of `tf.Variable` instances.

        Example:

        ```python
        export_archive = ExportArchive()
        export_archive.track(model)
        # Register an endpoint
        export_archive.add_endpoint(
            name="serve",
            fn=model.call,
            input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
        )
        # Save a variable collection
        export_archive.add_variable_collection(
            name="optimizer_variables", variables=model.optimizer.variables)
        export_archive.write_out("path/to/location")

        # Reload the object
        revived_object = tf.saved_model.load("path/to/location")
        # Retrieve the variables
        optimizer_variables = revived_object.optimizer_variables
        ```
        """
    if not isinstance(variables, (list, tuple, set)):
        raise ValueError(f"Expected `variables` to be a list/tuple/set. Received instead object of type '{type(variables)}'.")
    if not all((isinstance(v, tf.Variable) for v in variables)):
        raise ValueError(f'Expected all elements in `variables` to be `tf.Variable` instances. Found instead the following types: {list(set((type(v) for v in variables)))}')
    setattr(self, name, list(variables))