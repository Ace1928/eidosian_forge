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
def map_calibration(qubits, parameters, schedule):
    modified = False
    new_parameters = list(parameters)
    for i, parameter in enumerate(new_parameters):
        if not isinstance(parameter, ParameterExpression):
            continue
        if not (contained := (parameter.parameters & parameter_binds.mapping.keys())):
            continue
        for to_bind in contained:
            parameter = parameter.assign(to_bind, parameter_binds.mapping[to_bind])
        if not parameter.parameters:
            parameter = parameter.numeric()
            if isinstance(parameter, complex):
                raise TypeError(f"Calibration cannot use complex number: '{parameter}'")
        new_parameters[i] = parameter
        modified = True
    if modified:
        schedule.assign_parameters(parameter_binds.mapping)
    return ((qubits, tuple(new_parameters)), schedule)