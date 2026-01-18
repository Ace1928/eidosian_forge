import collections
import threading
import numpy as np
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import op_callbacks_common
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
@tf_export('debugging.disable_check_numerics')
def disable_check_numerics():
    """Disable the eager/graph unified numerics checking mechanism.

  This method can be used after a call to `tf.debugging.enable_check_numerics()`
  to disable the numerics-checking mechanism that catches infinity and NaN
  values output by ops executed eagerly or in tf.function-compiled graphs.

  This method is idempotent. Calling it multiple times has the same effect
  as calling it once.

  This method takes effect only on the thread in which it is called.
  """
    if not hasattr(_state, 'check_numerics_callback'):
        return
    try:
        op_callbacks.remove_op_callback(_state.check_numerics_callback.callback)
        delattr(_state, 'check_numerics_callback')
        logging.info('Disabled check-numerics callback in thread %s', threading.current_thread().name)
    except KeyError:
        pass