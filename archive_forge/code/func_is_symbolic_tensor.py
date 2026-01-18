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
def is_symbolic_tensor(tensor):
    """Returns whether a tensor is symbolic (from a TF graph) or an eager tensor.

  A Variable can be seen as either: it is considered symbolic
  when we are in a graph scope, and eager when we are in an eager scope.

  Args:
    tensor: A tensor instance to test.

  Returns:
    True for symbolic tensors, False for eager tensors.
  """
    if isinstance(tensor, tensor_lib.Tensor):
        return hasattr(tensor, 'graph')
    elif is_extension_type(tensor):
        component_tensors = nest.flatten(tensor, expand_composites=True)
        return any((hasattr(t, 'graph') for t in component_tensors))
    elif isinstance(tensor, variables.Variable):
        return getattr(tensor, '_keras_history', False) or not context.executing_eagerly()
    elif isinstance(tensor, tuple(_user_convertible_tensor_types)):
        tensor = ops.convert_to_tensor_or_composite(tensor)
        return is_symbolic_tensor(tensor)
    else:
        return False