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
@staticmethod
def from_qasm_str(qasm_str: str) -> 'QuantumCircuit':
    """Convert a string containing an OpenQASM 2.0 program to a :class:`.QuantumCircuit`.

        Args:
          qasm_str (str): A string containing an OpenQASM 2.0 program.
        Return:
          QuantumCircuit: The QuantumCircuit object for the input OpenQASM 2

        See also:
            :func:`.qasm2.loads`: the complete interface to the OpenQASM 2 importer.
        """
    from qiskit import qasm2
    return qasm2.loads(qasm_str, include_path=qasm2.LEGACY_INCLUDE_PATH, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS, custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL, strict=False)