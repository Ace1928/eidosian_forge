import collections
import copy
import numpy as np
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.util import nest
def get_reachable_from_inputs(inputs, targets=None):
    """Returns the set of tensors/ops reachable from `inputs`.

  Stops if all targets have been found (target is optional).

  Only valid in Symbolic mode, not Eager mode.

  Args:
    inputs: List of tensors.
    targets: List of tensors.

  Returns:
    A set of tensors reachable from the inputs (includes the inputs themselves).
  """
    inputs = nest.flatten(inputs, expand_composites=True)
    reachable = object_identity.ObjectIdentitySet(inputs)
    if targets:
        remaining_targets = object_identity.ObjectIdentitySet(nest.flatten(targets))
    queue = collections.deque(inputs)
    while queue:
        x = queue.pop()
        if isinstance(x, tuple(_user_convertible_tensor_types)):
            continue
        if isinstance(x, ops.Operation):
            outputs = x.outputs[:] or []
            outputs += x._control_outputs
        elif isinstance(x, variables.Variable):
            try:
                outputs = [x.op]
            except AttributeError:
                outputs = []
        elif tensor_util.is_tf_type(x):
            outputs = x.consumers()
        else:
            raise TypeError('Expected Operation, Variable, or Tensor, got ' + str(x))
        for y in outputs:
            if y not in reachable:
                reachable.add(y)
                if targets:
                    remaining_targets.discard(y)
                queue.appendleft(y)
        if targets and (not remaining_targets):
            return reachable
    return reachable