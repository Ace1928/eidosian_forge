from __future__ import annotations
import copy
import multiprocessing as mp
import typing
from collections import OrderedDict, defaultdict, namedtuple
from typing import (
import numpy as np
from qiskit._accelerate.quantum_circuit import CircuitData
from qiskit.exceptions import QiskitError
from qiskit.utils.multiprocessing import is_main_process
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.exceptions import CircuitError
from . import _classical_resource_map
from ._utils import sort_parameters
from .controlflow.builder import CircuitScopeInterface, ControlFlowBuilderBlock
from .controlflow.break_loop import BreakLoopOp, BreakLoopPlaceholder
from .controlflow.continue_loop import ContinueLoopOp, ContinueLoopPlaceholder
from .controlflow.for_loop import ForLoopOp, ForLoopContext
from .controlflow.if_else import IfElseOp, IfContext
from .controlflow.switch_case import SwitchCaseOp, SwitchContext
from .controlflow.while_loop import WhileLoopOp, WhileLoopContext
from .classical import expr
from .parameterexpression import ParameterExpression, ParameterValueType
from .quantumregister import QuantumRegister, Qubit, AncillaRegister, AncillaQubit
from .classicalregister import ClassicalRegister, Clbit
from .parametertable import ParameterReferences, ParameterTable, ParameterView
from .parametervector import ParameterVector
from .instructionset import InstructionSet
from .operation import Operation
from .register import Register
from .bit import Bit
from .quantumcircuitdata import QuantumCircuitData, CircuitInstruction
from .delay import Delay
def has_calibration_for(self, instruction: CircuitInstruction | tuple):
    """Return True if the circuit has a calibration defined for the instruction context. In this
        case, the operation does not need to be translated to the device basis.
        """
    if isinstance(instruction, CircuitInstruction):
        operation = instruction.operation
        qubits = instruction.qubits
    else:
        operation, qubits, _ = instruction
    if not self.calibrations or operation.name not in self.calibrations:
        return False
    qubits = tuple((self.qubits.index(qubit) for qubit in qubits))
    params = []
    for p in operation.params:
        if isinstance(p, ParameterExpression) and (not p.parameters):
            params.append(float(p))
        else:
            params.append(p)
    params = tuple(params)
    return (qubits, params) in self.calibrations[operation.name]