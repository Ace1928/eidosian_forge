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
@property
def preferred_init_points(self) -> list[float] | None:
    """The initial points for the parameters. Can be stored as initial guess in optimization.

        Returns:
            The initial values for the parameters, or None, if none have been set.
        """
    return None