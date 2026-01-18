import functools
import threading
from tensorflow.python import tf2
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as tracking
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
def call_context():
    """Returns currently active `CallContext`."""
    call_ctx = getattr(_call_context, 'call_context', None)
    if call_ctx is None:
        call_ctx = CallContext()
        _call_context.call_context = call_ctx
    return call_ctx