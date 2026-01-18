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
def build_gate_signature(self, gate):
    """Builds a QuantumGateSignature"""
    name = self.global_namespace[gate]
    params = []
    definition = gate.definition
    scope = self.current_scope()
    for num in range(len(gate.params) - len(definition.parameters)):
        param_name = f'{self.gate_parameter_prefix}_{num}'
        params.append(self._reserve_variable_name(ast.Identifier(param_name), scope))
    params += [self._register_variable(param, scope) for param in definition.parameters]
    quantum_arguments = [self._register_variable(qubit, scope, self._unique_name(f'{self.gate_qubit_prefix}_{i}', scope)) for i, qubit in enumerate(definition.qubits)]
    return ast.QuantumGateSignature(ast.Identifier(name), quantum_arguments, params or None)