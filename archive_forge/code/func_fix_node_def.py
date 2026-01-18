import collections
import pprint
import re
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as function_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import func_graph as func_graph_lib
from tensorflow.python.framework import function_def_to_graph as function_def_lib
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
def fix_node_def(node_def, functions, shared_name_suffix):
    """Replace functions calls and shared names in `node_def`."""
    if node_def.op in functions:
        node_def.op = functions[node_def.op].name
    for _, attr_value in node_def.attr.items():
        if attr_value.WhichOneof('value') == 'func':
            attr_value.func.name = functions[attr_value.func.name].name
        elif attr_value.WhichOneof('value') == 'list':
            for fn in attr_value.list.func:
                fn.name = functions[fn.name].name
    if node_def.op == 'HashTableV2':
        if 'use_node_name_sharing' not in node_def.attr or not node_def.attr['use_node_name_sharing'].b:
            node_def.attr['use_node_name_sharing'].b = True
            shared_name_suffix += '_{}'.format(ops.uid())
    op_def = op_def_registry.get(node_def.op)
    if op_def:
        attr = next((a for a in op_def.attr if a.name == 'shared_name'), None)
        if attr:
            shared_name = None
            if 'shared_name' in node_def.attr and node_def.attr['shared_name'].s:
                shared_name = node_def.attr['shared_name'].s
            elif attr.default_value.s:
                shared_name = compat.as_bytes(attr.default_value.s)
            if not shared_name:
                shared_name = compat.as_bytes(node_def.name)
            node_def.attr['shared_name'].s = shared_name + compat.as_bytes(shared_name_suffix)