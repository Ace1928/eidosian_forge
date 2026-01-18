from cirq import ops
from cirq.neutral_atoms import neutral_atom_devices
def is_native_neutral_atom_gate(gate: ops.Gate) -> bool:
    """Returns true if the gate is in the default neutral atom gateset."""
    return gate in neutral_atom_devices.neutral_atom_gateset()