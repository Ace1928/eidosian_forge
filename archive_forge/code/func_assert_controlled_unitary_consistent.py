from typing import Sequence, Optional, Union, Collection
from cirq import devices, ops, protocols
import numpy as np
def assert_controlled_unitary_consistent(gate: ops.Gate):
    """Checks that unitary of ControlledGate(gate) is consistent with gate.controlled()."""
    u_orig = protocols.unitary(ops.ControlledGate(gate))
    u_controlled = protocols.unitary(gate.controlled())
    np.testing.assert_allclose(u_orig, u_controlled, atol=1e-06, err_msg=f'Unitary for gate.controlled() is inconsistent for gate={gate!r}')