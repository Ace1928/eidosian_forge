from __future__ import annotations
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError, UnassignedReferenceError
from qiskit.pulse.instructions import instruction
@property
def ref_keys(self) -> tuple[str, ...]:
    """Returns unique key of the subroutine."""
    return self.operands