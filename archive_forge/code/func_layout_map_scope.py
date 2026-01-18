import collections
import contextlib
import re
import threading
import tensorflow.compat.v2 as tf
from keras.src.dtensor import dtensor_api as dtensor
from keras.src.dtensor import lazy_variable
from keras.src.dtensor import utils
from keras.src.engine import base_layer
from tensorflow.python.util.tf_export import keras_export
@contextlib.contextmanager
def layout_map_scope(layout_map):
    """Apply the layout to all the tf.Variables created under the scope.

    Create a scope that all the tf.Variable created under this scope
    will be lazily inited, and initialized later on with proper layout when the
    object path in the model is stable/finalized.

    Note that the layout mapping will use the object/attribute names as the key
    to map the variable against the layout.

    For subclassed models, the full object/attribute name is used as the key.
    For Functional/Sequential models, since the layers within the model do not
    get assigned to a meaningful attribute, we use `layer.name` as the key for
    the layer, followed by the attribute name. Keras ensures name uniqueness
    among the layers in all Functional/Sequential models.

    See the following examples that show the variable object names
    for different Keras model types:

    ```python
    layout_map = layout_map_lib.LayoutMap(mesh=self.mesh)
    layout_map['d1.kernel'] = layout_1
    layout_map['d1.bias'] = layout_2
    layout_map['d2.kernel'] = layout_3
    layout_map['d2.bias'] = layout_4

    ## Subclassed model
    class SubclassModel(tf.keras.Model):

      def __init__(self, name=None):
        super().__init__(name=name)
        self.d1 = tf.keras.layers.Dense(1000)
        self.d2 = tf.keras.layers.Dense(1000)

      def call(self, inputs):
        x = self.d1(inputs)
        return self.d2(x)

    with layout_map_scope(layout_map):
      model = SubclassModel()
    # Triggering the creation of weights within or outside of the scope works
    inputs = tf.zeros((10, 10))
    results = model(inputs)

    model.d1.kernel.layout == layout_1
    model.d1.bias.layout == layout_2
    model.d2.kernel.layout == layout_3
    model.d2.bias.layout == layout_4

    ## Functional model
    with layout_map_scope(layout_map):
      inputs = tf.keras.Input((10,), batch_size=10)
      x = tf.keras.layers.Dense(20, name='d1')(inputs)
      output = tf.keras.layers.Dense(30, name='d2')(x)

      model = tf.keras.Model(inputs, output)

    d1 = model.layers[1]
    d2 = model.layers[2]

    d1.kernel.layout == layout_1
    d1.bias.layout == layout_2
    d1.kernel.layout == layout_3
    d1.bias.layout == layout_4

    ## Sequential model
    with layout_map_scope(layout_map):
      model = tf.keras.Sequential([
          tf.keras.layers.Dense(20, name='d1', input_shape=(10,)),
          tf.keras.layers.Dense(30, name='d2')
      ])

    d1 = model.layers[0]
    d2 = model.layers[1]

    d1.kernel.layout == layout_1
    d1.bias.layout == layout_2
    d1.kernel.layout == layout_3
    d1.bias.layout == layout_4
    ```

    Args:
      layout_map: a LayoutMap which contains the variable_object_path (string)
        -> Layout. When a layout is not found for the variable, a default all
        replicated layout will be created for the variable.

    Yields:
      A context that will lazily initialize all `tf.Variable` objects
      within the model, with their attributed layouts.
    """
    previous_layout_map = get_current_layout_map()
    global _LAYOUT_MAP
    _LAYOUT_MAP.layout_map = layout_map
    with lazy_variable.lazy_init_scope():
        try:
            yield
        finally:
            _LAYOUT_MAP.layout_map = previous_layout_map