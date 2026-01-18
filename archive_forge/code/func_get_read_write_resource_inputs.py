from tensorflow.python.framework import dtypes
from tensorflow.python.util import object_identity
def get_read_write_resource_inputs(op):
    """Returns a tuple of resource reads, writes in op.inputs.

  Args:
    op: Operation

  Returns:
    A 2-tuple of ObjectIdentitySets, the first entry containing read-only
    resource handles and the second containing read-write resource handles in
    `op.inputs`.
  """
    reads = object_identity.ObjectIdentitySet()
    writes = object_identity.ObjectIdentitySet()
    if op.type in RESOURCE_READ_OPS:
        reads.update((t for t in op.inputs if t.dtype == dtypes.resource))
        return (reads, writes)
    try:
        read_only_input_indices = op.get_attr(READ_ONLY_RESOURCE_INPUTS_ATTR)
    except ValueError:
        writes.update((t for t in op.inputs if t.dtype == dtypes.resource))
        return (reads, writes)
    read_only_index = 0
    for i, t in enumerate(op.inputs):
        if op.inputs[i].dtype != dtypes.resource:
            continue
        if read_only_index < len(read_only_input_indices) and i == read_only_input_indices[read_only_index]:
            reads.add(op.inputs[i])
            read_only_index += 1
        else:
            writes.add(op.inputs[i])
    return (reads, writes)