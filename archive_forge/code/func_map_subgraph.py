from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import object_identity
def map_subgraph(init_tensor, sources, disallowed_placeholders, visited_ops, op_outputs, add_sources):
    """Walk a Graph and capture the subgraph between init_tensor and sources.

  Note: This function mutates visited_ops and op_outputs.

  Args:
    init_tensor:  A Tensor or Operation where the subgraph terminates.
    sources:  A set of Tensors where subgraph extraction should stop.
    disallowed_placeholders: An optional set of ops which may not appear in the
      lifted graph. Defaults to all placeholders.
    visited_ops: A set of operations which were visited in a prior pass.
    op_outputs: A defaultdict containing the outputs of an op which are to be
      copied into the new subgraph.
    add_sources: A boolean indicating whether placeholders which are not in
      sources should be allowed.

  Returns:
    The set of placeholders upon which init_tensor depends and are not in
    sources.

  Raises:
    UnliftableError: if init_tensor depends on a placeholder which is not in
      sources and add_sources is False.
  """
    ops_to_visit = [_as_operation(init_tensor)]
    extra_sources = object_identity.ObjectIdentitySet()
    while ops_to_visit:
        op = ops_to_visit.pop()
        if op in visited_ops:
            continue
        visited_ops.add(op)
        should_raise = False
        if disallowed_placeholders is not None and op in disallowed_placeholders:
            should_raise = True
        elif op.type == 'Placeholder':
            if disallowed_placeholders is None and (not add_sources):
                should_raise = True
            extra_sources.update(op.outputs)
        if should_raise:
            raise UnliftableError('Unable to lift tensor %s because it depends transitively on placeholder %s via at least one path, e.g.: %s' % (repr(init_tensor), repr(op), show_path(op, init_tensor, sources)))
        for inp in graph_inputs(op):
            op_outputs[inp].add(op)
            if inp not in visited_ops and inp not in (sources or extra_sources):
                ops_to_visit.append(inp)
    return extra_sources