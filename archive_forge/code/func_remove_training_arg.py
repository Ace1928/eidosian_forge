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
def remove_training_arg(index, args, kwargs):
    if index is None or index < 0 or len(args) <= index:
        kwargs.pop('training', None)
    else:
        args.pop(index)