import functools
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util.tf_export import tf_export
def get_host_dtensor():
    if original_layout.mesh.device_type().upper() != 'CPU':
        if context.executing_eagerly():
            host_dtensor = api.pack(api.unpack(dvariable.read_value()), host_layout)
        else:
            host_dtensor = api.copy_to_mesh(dvariable.read_value(), host_layout)
    else:
        host_dtensor = dvariable.read_value()
    return math_ops.cast(host_dtensor, dtypes.bfloat16) if self.should_cast(host_dtensor) else host_dtensor