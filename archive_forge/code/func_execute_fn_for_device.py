from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.execute_fn_for_device', v1=[])
def execute_fn_for_device(device_branch_fns, default_fn, name='execute_fn'):
    """Executes one of the provided callables based on the device placement.

  This API is used when the implementations for high level function depend on
  the underlying device placement. It takes a dictionary of device type to
  callables. The device type includes "CPU", "GPU", "TPU", etc. When the type of
  the device where to run this op matches the key in 'device_branch_fns',
  the corresponding callable is executed, falling back to 'default_fn' if none
  matches.

  **Example:**
  ```python
  def f1(): return tf.constant(1)
  def f2(): return tf.constant(2)
  r = tf.execute_fn_for_device({"CPU": f1, "GPU": f2}, default_fn=f1)
  ```
  'r' is evaluated as 1 when it runs on CPU, 2 running on GPU, 1 running on
  any other device types.


  Args:
    device_branch_fns: a dictionary of device types to the callables. Each
      callable must return a matching structure of tensors.
    default_fn: fallback callable when the underlying device does not match any
      key in the 'device_branch_fns'.
    name: A name for this operation (optional).

  Returns:
    The tensors returned by the callable identified by device type during
    execution, or those returned by 'default_fn' if no key matches.
  """
    is_in_xla = util.GraphOrParentsInXlaContext(ops.get_default_graph())
    if is_in_xla:
        return default_fn()
    device_branch_fns_upper = {k.upper(): v for k, v in device_branch_fns.items()}
    branch_fns = list(device_branch_fns_upper.values())
    devices = list(device_branch_fns_upper.keys())
    device_index = gen_functional_ops.device_index(device_names=devices)
    return _indexed_case_helper(branch_fns, default_fn, device_index, name, lower_using_switch_merge=False)