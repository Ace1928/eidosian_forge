from __future__ import annotations
import typing
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
import numpy
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit import Instruction, Parameter, ParameterVector, ParameterExpression
from qiskit.exceptions import QiskitError
from ..blueprintcircuit import BlueprintCircuit
@entanglement_blocks.setter
def entanglement_blocks(self, blocks: QuantumCircuit | list[QuantumCircuit] | Instruction | list[Instruction]) -> None:
    """Set the blocks in the entanglement layers.

        Args:
            blocks: The new blocks for the entanglement layers.
        """
    if not isinstance(blocks, (list, numpy.ndarray)):
        blocks = [blocks]
    self._invalidate()
    self._entanglement_blocks = [self._convert_to_block(block) for block in blocks]