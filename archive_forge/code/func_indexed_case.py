import collections
from tensorflow.core.framework import types_pb2
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def indexed_case(branch_index, branch_fns, name='indexed_case', lower_using_switch_merge=None):
    """Like conv_v2, except emits a Case op instead of an If."""
    if isinstance(branch_index, int):
        raise TypeError('branch_index must not be a Python int', branch_index)
    with ops.name_scope(name) as scope:
        branch_names = [util.unique_fn_name(scope, 'branch{}'.format(b)) for b in range(len(branch_fns))]
        add_control_dependencies = ops.get_default_graph()._add_control_dependencies
        branch_index = ops.convert_to_tensor(branch_index, name='branch_index')
        branch_graphs = []
        for branch_name, branch_fn in zip(branch_names, branch_fns):
            branch_graphs.append(func_graph_module.func_graph_from_py_func(branch_name, branch_fn, [], {}, func_graph=util.CondBranchFuncGraph(branch_name, collections=ops.get_default_graph()._collections), add_control_dependencies=add_control_dependencies, op_return_value=branch_index))
        verify_captures(_CASE, branch_graphs)
        return _build_case(branch_index, branch_graphs, [g.external_captures for g in branch_graphs], name=scope, lower_using_switch_merge=lower_using_switch_merge)