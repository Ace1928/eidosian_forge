import collections
import re
import io
import itertools
import numbers
from os.path import dirname, join, abspath
from typing import Iterable, List, Sequence, Union
from qiskit.circuit import (
from qiskit.circuit.bit import Bit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.controlflow import (
from qiskit.circuit.library import standard_gates
from qiskit.circuit.register import Register
from qiskit.circuit.tools import pi_check
from . import ast
from .experimental import ExperimentalFeatures
from .exceptions import QASM3ExporterError
from .printer import BasicPrinter
def build_delay(self, instruction: CircuitInstruction) -> ast.QuantumDelay:
    """Build a built-in delay statement."""
    if instruction.clbits:
        raise QASM3ExporterError(f'Found a delay instruction acting on classical bits: {instruction}')
    duration_value, unit = (instruction.operation.duration, instruction.operation.unit)
    if unit == 'ps':
        duration = ast.DurationLiteral(1000 * duration_value, ast.DurationUnit.NANOSECOND)
    else:
        unit_map = {'ns': ast.DurationUnit.NANOSECOND, 'us': ast.DurationUnit.MICROSECOND, 'ms': ast.DurationUnit.MILLISECOND, 's': ast.DurationUnit.SECOND, 'dt': ast.DurationUnit.SAMPLE}
        duration = ast.DurationLiteral(duration_value, unit_map[unit])
    return ast.QuantumDelay(duration, [self._lookup_variable(qubit) for qubit in instruction.qubits])