from tensorflow.python.eager import context
from tensorflow.python.eager import execute
def invoke_op_callbacks(op_type, inputs, attrs, outputs, op_name=None, graph=None):
    """Invoke the callbacks that exist in the current scope (if any).

  If no callbacks are present in the current scope, this method returns
  immediately.

  Args:
    op_type: Type of the operation (e.g., "MatMul").
    inputs: Input tensors to the op. These are `EagerTensor`s in the case of
      eager execution of ops or `FuncGraph`s, and are non-eager `Tensor`s in the
      case of graph construction.
    attrs: Attributes of the op, as `tuple` of alternating keys and values.
    outputs: Output tensors from the op. These are `EagerTensor`s in the case of
      eager execution and are non-eager `Tensor`s in the case of graph
      construction.
    op_name: Name of the op. Applicable if and only if this method is invoked
      due to the graph construction of an op or the eager execution of a
      `FuncGraph`.
    graph: The graph involved (if any).
      - In the case if the eager execution of an op or FuncGraph, this is
        `None`.
      - In the case of the graph construction of an op, this is the `tf.Graph`
        object being built.

  Returns:
    `None`, or a `list` or `tuple` of output tenors that will override the
    original (input) `outputs`.
  """
    ctx = context.context()
    if ctx.op_callbacks:
        ctx.invoking_op_callbacks = True
        try:
            if isinstance(attrs, dict):
                attrs_list = []
                for key in attrs:
                    attrs_list.append(key)
                    attrs_list.append(attrs[key])
                attrs_tuple = tuple(attrs_list)
            else:
                attrs_tuple = attrs
            new_outputs = outputs
            for callback in ctx.op_callbacks:
                new_outputs = callback(op_type, inputs, attrs_tuple, new_outputs, op_name=op_name, graph=graph)
                if new_outputs is not None and len(new_outputs) != len(outputs):
                    raise ValueError(f'The op callback returned {len(new_outputs)} tensors, which does not match the original number of outputs of op {op_name} ({len(outputs)}).')
            return new_outputs
        finally:
            ctx.invoking_op_callbacks = False
    else:
        return outputs