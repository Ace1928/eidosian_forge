import functools
import operator
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import imperative_grad
from tensorflow.python.eager import tape
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.record_gradient', v1=[])
def record_gradient(op_name, inputs, attrs, outputs):
    """Explicitly record the gradient for a given op.

  Args:
    op_name: The op name as listed in the `OpDef` for the op.
    inputs: A list of tensor inputs to the op.
    attrs: The op attributes as a flattened list of alternating attribute names
      and attribute values.
    outputs: A list of tensor outputs from the op.
  """
    pywrap_tfe.TFE_Py_RecordGradient(op_name, inputs, attrs, outputs, ops.get_name_scope())