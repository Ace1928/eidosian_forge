from __future__ import annotations
import itertools
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Type
from qiskit.circuit.quantumcircuit import ClbitSpecifier, QubitSpecifier
from qiskit.circuit.delay import Delay
from qiskit.circuit.measure import Measure
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.utils.deprecation import deprecate_func
def pad_with_delays(qubits: Iterable[QubitSpecifier], until, unit) -> None:
    """Pad idle time-slots in ``qubits`` with delays in ``unit`` until ``until``."""
    for q in qubits:
        if qubit_stop_times[q] < until:
            idle_duration = until - qubit_stop_times[q]
            new_dag.apply_operation_back(Delay(idle_duration, unit), (q,), check=False)