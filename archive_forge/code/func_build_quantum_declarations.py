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
def build_quantum_declarations(self):
    """Return a list of AST nodes declaring all the qubits in the current scope, and all the
        alias declarations for these qubits."""
    scope = self.global_scope(assert_=True)
    if scope.circuit.layout is not None:
        for i, bit in enumerate(scope.circuit.qubits):
            scope.symbol_map[bit] = ast.Identifier(f'${i}')
        return []
    if any((len(scope.circuit.find_bit(q).registers) > 1 for q in scope.circuit.qubits)):
        if not self.allow_aliasing:
            raise QASM3ExporterError("Some quantum registers in this circuit overlap and need aliases to express, but 'allow_aliasing' is false.")
        qubits = [ast.QuantumDeclaration(self._register_variable(qubit, scope, self._unique_name(f'{self.loose_qubit_prefix}{i}', scope))) for i, qubit in enumerate(scope.circuit.qubits)]
        return qubits + self.build_aliases(scope.circuit.qregs)
    loose_qubits = [ast.QuantumDeclaration(self._register_variable(qubit, scope, self._unique_name(f'{self.loose_qubit_prefix}{i}', scope))) for i, qubit in enumerate(scope.circuit.qubits) if not scope.circuit.find_bit(qubit).registers]
    registers = []
    for register in scope.circuit.qregs:
        name = self._register_variable(register, scope)
        for i, bit in enumerate(register):
            scope.symbol_map[bit] = ast.SubscriptedIdentifier(name.string, ast.IntegerLiteral(i))
        registers.append(ast.QuantumDeclaration(name, ast.Designator(ast.IntegerLiteral(len(register)))))
    return loose_qubits + registers