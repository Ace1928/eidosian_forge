from __future__ import annotations
from qiskit.circuit import ParameterExpression
from qiskit.pulse.channels import MemorySlot, RegisterSlot, AcquireChannel
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.instructions.instruction import Instruction
@property
def reg_slot(self) -> RegisterSlot:
    """The fast-access register slot which will store the classified readout result for
        fast-feedback computation.
        """
    return self.operands[3]