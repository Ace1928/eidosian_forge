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
def build_quantum_instructions(self, instructions):
    """Builds a list of call statements"""
    ret = []
    for instruction in instructions:
        if isinstance(instruction.operation, ForLoopOp):
            ret.append(self.build_for_loop(instruction))
            continue
        if isinstance(instruction.operation, WhileLoopOp):
            ret.append(self.build_while_loop(instruction))
            continue
        if isinstance(instruction.operation, IfElseOp):
            ret.append(self.build_if_statement(instruction))
            continue
        if isinstance(instruction.operation, SwitchCaseOp):
            ret.extend(self.build_switch_statement(instruction))
            continue
        if isinstance(instruction.operation, Gate):
            nodes = [self.build_gate_call(instruction)]
        elif isinstance(instruction.operation, Barrier):
            operands = [self._lookup_variable(operand) for operand in instruction.qubits]
            nodes = [ast.QuantumBarrier(operands)]
        elif isinstance(instruction.operation, Measure):
            measurement = ast.QuantumMeasurement([self._lookup_variable(operand) for operand in instruction.qubits])
            qubit = self._lookup_variable(instruction.clbits[0])
            nodes = [ast.QuantumMeasurementAssignment(qubit, measurement)]
        elif isinstance(instruction.operation, Reset):
            nodes = [ast.QuantumReset(self._lookup_variable(operand)) for operand in instruction.qubits]
        elif isinstance(instruction.operation, Delay):
            nodes = [self.build_delay(instruction)]
        elif isinstance(instruction.operation, BreakLoopOp):
            nodes = [ast.BreakStatement()]
        elif isinstance(instruction.operation, ContinueLoopOp):
            nodes = [ast.ContinueStatement()]
        else:
            nodes = [self.build_subroutine_call(instruction)]
        if instruction.operation.condition is None:
            ret.extend(nodes)
        else:
            body = ast.ProgramBlock(nodes)
            ret.append(ast.BranchingStatement(self.build_expression(_lift_condition(instruction.operation.condition)), body))
    return ret