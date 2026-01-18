from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import op_selector
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def grad_wrapper(*wrapper_args, variables=None):
    """Wrapper function to accomodate lack of kwargs in graph mode custom_gradient."""

    @custom_gradient
    def inner_recompute_grad(*dresult):
        """Nested custom gradient function for computing grads in reverse and forward mode autodiff."""
        with backprop.GradientTape() as t:
            id_args = nest.map_structure(gen_array_ops.identity, args)
            assert len(dresult) >= 1
            if not context.executing_eagerly():
                elem = math_ops.reduce_max(array_ops.reshape(dresult[0], [-1])[:1])
                elem_bool = math_ops.cast(elem, dtypes.bool)
                dresult_dep = array_ops.where_v2(elem_bool == elem_bool, 0.0, float('nan'))
                id_args = nest.map_structure(lambda x: x + math_ops.cast(dresult_dep, x.dtype), id_args)
            t.watch(id_args)
            if variables is not None:
                t.watch(variables)
            with variable_scope.variable_scope(current_var_scope):
                recomputed_result = f(*id_args, **kwargs)
        kw_vars = []
        if variables is not None:
            kw_vars = list(variables)
        grads = t.gradient(recomputed_result, list(id_args) + kw_vars, output_gradients=dresult, unconnected_gradients=UnconnectedGradients.ZERO)

        def transpose(*t_args, **t_kwargs):
            """Gradient function calculation for forward mode autodiff."""
            raise NotImplementedError('recompute_grad tried to transpose grad of {}. Consider not using recompute_grad in forward modeautodiff'.format(f.__name__))
        return ((grads[:len(id_args)], grads[len(id_args):]), transpose)
    return inner_recompute_grad(*wrapper_args)