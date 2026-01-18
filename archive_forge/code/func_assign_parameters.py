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
def assign_parameters(self, parameters: Mapping[Parameter, ParameterExpression | float] | Sequence[ParameterExpression | float], inplace: bool=False, **kwargs) -> QuantumCircuit | None:
    """Assign parameters to the n-local circuit.

        This method also supports passing a list instead of a dictionary. If a list
        is passed, the list must have the same length as the number of unbound parameters in
        the circuit. The parameters are assigned in the order of the parameters in
        :meth:`ordered_parameters`.

        Returns:
            A copy of the NLocal circuit with the specified parameters.

        Raises:
            AttributeError: If the parameters are given as list and do not match the number
                of parameters.
        """
    if parameters is None or len(parameters) == 0:
        return self
    if not self._is_built:
        self._build()
    return super().assign_parameters(parameters, inplace=inplace, **kwargs)