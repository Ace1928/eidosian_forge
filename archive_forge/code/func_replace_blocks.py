from __future__ import annotations
import typing
from abc import ABC, abstractmethod
from qiskit.circuit.instruction import Instruction
@abstractmethod
def replace_blocks(self, blocks: typing.Iterable[QuantumCircuit]) -> ControlFlowOp:
    """Replace blocks and return new instruction.
        Args:
            blocks: Tuple of QuantumCircuits to replace in instruction.

        Returns:
            New ControlFlowOp with replaced blocks.
        """