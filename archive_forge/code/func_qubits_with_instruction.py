from __future__ import annotations
import functools
import warnings
from collections import defaultdict
from collections.abc import Iterable, Callable
from qiskit import circuit
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.calibration_entries import (
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ScheduleBlock
def qubits_with_instruction(self, instruction: str | circuit.instruction.Instruction) -> list[int | tuple[int, ...]]:
    """Return a list of the qubits for which the given instruction is defined. Single qubit
        instructions return a flat list, and multiqubit instructions return a list of ordered
        tuples.

        Args:
            instruction: The name of the circuit instruction.

        Returns:
            Qubit indices which have the given instruction defined. This is a list of tuples if the
            instruction has an arity greater than 1, or a flat list of ints otherwise.

        Raises:
            PulseError: If the instruction is not found.
        """
    instruction = _get_instruction_string(instruction)
    if instruction not in self._map:
        return []
    return [qubits[0] if len(qubits) == 1 else qubits for qubits in sorted(self._map[instruction].keys())]