from __future__ import annotations
from qiskit.circuit import ParameterExpression
from qiskit.pulse.channels import MemorySlot, RegisterSlot, AcquireChannel
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.instructions.instruction import Instruction
@property
def mem_slot(self) -> MemorySlot:
    """The classical memory slot which will store the classified readout result."""
    return self.operands[2]