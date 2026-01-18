import threading
from tensorflow.python import tf2
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.layers import advanced_activations
from tensorflow.python.keras.layers import convolutional
from tensorflow.python.keras.layers import convolutional_recurrent
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import dense_attention
from tensorflow.python.keras.layers import embeddings
from tensorflow.python.keras.layers import merge
from tensorflow.python.keras.layers import pooling
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.keras.layers import rnn_cell_wrapper_v2
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect as inspect
def populate_deserializable_objects():
    """Populates dict ALL_OBJECTS with every built-in layer.
  """
    global LOCAL
    if not hasattr(LOCAL, 'ALL_OBJECTS'):
        LOCAL.ALL_OBJECTS = {}
        LOCAL.GENERATED_WITH_V2 = None
    if LOCAL.ALL_OBJECTS and LOCAL.GENERATED_WITH_V2 == tf2.enabled():
        return
    LOCAL.ALL_OBJECTS = {}
    LOCAL.GENERATED_WITH_V2 = tf2.enabled()
    base_cls = base_layer.Layer
    generic_utils.populate_dict_with_module_objects(LOCAL.ALL_OBJECTS, ALL_MODULES, obj_filter=lambda x: inspect.isclass(x) and issubclass(x, base_cls))
    if tf2.enabled():
        generic_utils.populate_dict_with_module_objects(LOCAL.ALL_OBJECTS, ALL_V2_MODULES, obj_filter=lambda x: inspect.isclass(x) and issubclass(x, base_cls))
    from tensorflow.python.keras import models
    LOCAL.ALL_OBJECTS['Input'] = input_layer.Input
    LOCAL.ALL_OBJECTS['InputSpec'] = input_spec.InputSpec
    LOCAL.ALL_OBJECTS['Functional'] = models.Functional
    LOCAL.ALL_OBJECTS['Model'] = models.Model
    LOCAL.ALL_OBJECTS['Sequential'] = models.Sequential
    LOCAL.ALL_OBJECTS['add'] = merge.add
    LOCAL.ALL_OBJECTS['subtract'] = merge.subtract
    LOCAL.ALL_OBJECTS['multiply'] = merge.multiply
    LOCAL.ALL_OBJECTS['average'] = merge.average
    LOCAL.ALL_OBJECTS['maximum'] = merge.maximum
    LOCAL.ALL_OBJECTS['minimum'] = merge.minimum
    LOCAL.ALL_OBJECTS['concatenate'] = merge.concatenate
    LOCAL.ALL_OBJECTS['dot'] = merge.dot