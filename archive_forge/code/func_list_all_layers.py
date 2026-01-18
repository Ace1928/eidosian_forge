import itertools
import threading
import types
from tensorflow.python.eager import context
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.util import tf_decorator
def list_all_layers(obj):
    if isinstance(obj, training_lib.Model):
        return obj.layers
    else:
        return list(obj._flatten_layers(include_self=False, recursive=False))