from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.operators import variables
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import nest
def ret(self, value, did_return):
    """Marks a value as returned from the function guarded by the scope."""
    del did_return
    if isinstance(value, variables.UndefinedReturnValue):
        return None
    if self.use_auto_deps:
        self._return_value_marked = True
        if value is None:
            return None

        def _mark_return_if_tensor(t):
            if tensor_util.is_tf_type(t):
                return self.autodeps_scope.mark_as_return(t)
            return t
        value = nest.map_structure(_mark_return_if_tensor, value)
    return value