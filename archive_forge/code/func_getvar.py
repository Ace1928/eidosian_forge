import collections
import hashlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
def getvar(self, getter, name, shape=None, dtype=None, initializer=None, reuse=None, trainable=True, collections=None, use_resource=None, **kwargs):
    """A custom variable getter."""
    with self._outer_graph.as_default():
        var = self._vscope.get_variable(vs._get_default_variable_store(), name, shape=shape, dtype=dtype, initializer=initializer, reuse=reuse, trainable=trainable, collections=collections, use_resource=use_resource)
        self.extra_vars.append(var)
        if isinstance(var, resource_variable_ops.BaseResourceVariable) and self._capture_resource_var_by_value:
            return var.value()
        return var