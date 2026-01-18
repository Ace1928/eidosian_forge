from cirq import ops
from cirq.neutral_atoms import neutral_atom_devices
def is_native_neutral_atom_op(operation: ops.Operation) -> bool:
    """Returns true if the operation is in the default neutral atom gateset."""
    return operation in neutral_atom_devices.neutral_atom_gateset()