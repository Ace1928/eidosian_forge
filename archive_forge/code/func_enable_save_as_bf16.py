import collections
from typing import Dict, List, Union
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import mesh_util
from tensorflow.python.eager import context
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.enable_save_as_bf16', v1=[])
def enable_save_as_bf16(variables: List[tf_variables.Variable]):
    """Allows float32 DVariables to be checkpointed and restored as bfloat16.

  The method only affects the DVariable part inside the model and leaves
  non-DTensor Variables/Tensors untouched.

  Args:
    variables: A list of tf.Variable to be enabled with bfloat16 save/restore.
      Only has effect on DTensor Variables as they go through d_variables with
      DTensor Specific logis.
  """
    for v in variables:
        if isinstance(v, d_variable.DVariable):
            v.save_as_bf16 = True