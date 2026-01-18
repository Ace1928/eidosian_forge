import re
from tensorflow.python import tf2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
@property
def name_scope(self):
    """Returns a `tf.name_scope` instance for this class."""
    if tf2.enabled():
        return self._name_scope
    else:
        return ops.name_scope(self._scope_name, skip_on_eager=False)