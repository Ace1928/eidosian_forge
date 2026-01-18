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
@tf_contextlib.contextmanager
def keras_option_scope(save_traces):
    previous_value = _save_options_context.save_traces
    try:
        _save_options_context.save_traces = save_traces
        yield
    finally:
        _save_options_context.save_traces = previous_value