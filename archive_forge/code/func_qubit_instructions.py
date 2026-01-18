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
def qubit_instructions(self, qubits: int | Iterable[int]) -> list[str]:
    """Return a list of the instruction names that are defined by the backend for the given
        qubit or qubits.

        Args:
            qubits: A qubit index, or a list or tuple of indices.

        Returns:
            All the instructions which are defined on the qubits.

            For 1 qubit, all the 1Q instructions defined. For multiple qubits, all the instructions
            which apply to that whole set of qubits (e.g. ``qubits=[0, 1]`` may return ``['cx']``).
        """
    if _to_tuple(qubits) in self._qubit_instructions:
        return list(self._qubit_instructions[_to_tuple(qubits)])
    return []