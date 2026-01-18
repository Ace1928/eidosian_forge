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
def build_switch_statement(self, instruction: CircuitInstruction) -> Iterable[ast.Statement]:
    """Build a :obj:`.SwitchCaseOp` into a :class:`.ast.SwitchStatement`."""
    real_target = self.build_expression(expr.lift(instruction.operation.target))
    global_scope = self.global_scope()
    target = self._reserve_variable_name(ast.Identifier(self._unique_name('switch_dummy', global_scope)), global_scope)
    self._global_classical_declarations.append(ast.ClassicalDeclaration(ast.IntType(), target, None))
    if ExperimentalFeatures.SWITCH_CASE_V1 in self.experimental:

        def case(values, case_block):
            values = [ast.DefaultCase() if v is CASE_DEFAULT else self.build_integer(v) for v in values]
            self.push_scope(case_block, instruction.qubits, instruction.clbits)
            case_body = self.build_program_block(case_block.data)
            self.pop_scope()
            return (values, case_body)
        return [ast.AssignmentStatement(target, real_target), ast.SwitchStatementPreview(target, (case(values, block) for values, block in instruction.operation.cases_specifier()))]
    cases = []
    default = None
    for values, block in instruction.operation.cases_specifier():
        self.push_scope(block, instruction.qubits, instruction.clbits)
        case_body = self.build_program_block(block.data)
        self.pop_scope()
        if CASE_DEFAULT in values:
            default = case_body
            continue
        cases.append(([self.build_integer(value) for value in values], case_body))
    return [ast.AssignmentStatement(target, real_target), ast.SwitchStatement(target, cases, default=default)]