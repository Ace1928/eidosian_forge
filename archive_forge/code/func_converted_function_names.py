import collections
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
@property
def converted_function_names(self):
    """Map from original to new function names.

    In order to avoid conflicts (two functions with the same name, one converted
    and one not), we need to change the name of every converted function to
    something that is hopefully unique.

    Returns:
      Map from original to new suggested function names.
    """
    if self._converted_function_names is None:
        parsed_names = []
        for name in self.functions:
            elements = name.rsplit('_', 1)
            if len(elements) == 2 and elements[1].isnumeric():
                parsed_names.append((int(elements[1]), elements[0], name))
            else:
                parsed_names.append((-1, name, name))
        self._converted_function_names = {name: '{}_frozen_{}'.format(base_name, ops.uid()) for _, base_name, name in sorted(parsed_names)}
    return self._converted_function_names