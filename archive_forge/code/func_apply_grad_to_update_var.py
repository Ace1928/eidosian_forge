import abc
import contextlib
import functools
import warnings
from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.saved_model import revived_types
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
def apply_grad_to_update_var(var, grad):
    """Apply gradient to variable."""
    if isinstance(var, tensor.Tensor):
        raise NotImplementedError('Trying to update a Tensor ', var)
    apply_kwargs = {}
    if isinstance(grad, indexed_slices.IndexedSlices):
        if var.constraint is not None:
            raise RuntimeError('Cannot use a constraint function on a sparse variable.')
        if 'apply_state' in self._sparse_apply_args:
            apply_kwargs['apply_state'] = apply_state
        return self._resource_apply_sparse_duplicate_indices(grad.values, var, grad.indices, **apply_kwargs)
    if 'apply_state' in self._dense_apply_args:
        apply_kwargs['apply_state'] = apply_state
    update_op = self._resource_apply_dense(grad, var, **apply_kwargs)
    if var.constraint is not None:
        with ops.control_dependencies([update_op]):
            return var.assign(var.constraint(var))
    else:
        return update_op