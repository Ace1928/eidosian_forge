import re
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
def graph_to_function_def(graph, operations, inputs, outputs, out_names=None):
    """Returns `graph` as a `FunctionDef` protocol buffer.

  This method creates a [`FunctionDef`](
  https://www.tensorflow.org/code/tensorflow/core/framework/function.proto)
  protocol buffer that contains all the ops in `operations`.  The
  operations become the body of the function.

  The arguments `inputs` and `outputs` will be listed as the inputs
  and outputs tensors of the function.  They must be lists of
  tensors present in the graph.  The lists can optionally be empty.

  Args:
    graph: Graph.
    operations: the operations to put in the function. Must be a subset of
     the operations in the graph.
    inputs: List of tensors. Inputs to the function.
    outputs: List of tensors. Outputs of the function.
    out_names: Optional list of string names for the outputs.

  Returns:
    A FunctionDef protocol buffer.

  Raises:
    ValueError: if out_names is specified and the wrong length.
  """
    func = function_pb2.FunctionDef()
    func.signature.name = '_'
    used_names = set()
    func.signature.input_arg.extend([_tensor_to_argdef(i, used_names=used_names) for i in inputs])
    initial_dict = {}
    for o, m in zip(inputs, func.signature.input_arg):
        initial_dict[o.name] = m.name
    if out_names is None:
        used_names = set()
        func.signature.output_arg.extend([_tensor_to_argdef(o, used_names=used_names) for o in outputs])
    elif len(outputs) != len(out_names):
        raise ValueError(f'out_names must be either empty or equal in size to outputs. len(out_names) = {len(out_names)} len(outputs) = {len(outputs)}')
    elif len(out_names) != len(set(out_names)):
        raise ValueError(f'Must not have duplicates in out_names. Received: {out_names}')
    else:
        func.signature.output_arg.extend([_tensor_to_argdef(o, name=n) for o, n in zip(outputs, out_names)])
    func_arg_placeholders = set((i.name for i in inputs))
    input_dict = _create_input_dict(graph, func_arg_placeholders, initial_value=initial_dict)
    for op in operations:
        if _is_in_placeholders(op, func_arg_placeholders):
            continue
        _add_op_node(op, func, input_dict)
    if out_names is None:
        for index, o in enumerate(outputs):
            k = func.signature.output_arg[index].name
            func.ret[k] = input_dict[o.name]
    else:
        for o, n in zip(outputs, out_names):
            func.ret[n] = input_dict[o.name]
    return func