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
def cs(self, control_qubit: QubitSpecifier, target_qubit: QubitSpecifier, label: str | None=None, ctrl_state: str | int | None=None) -> InstructionSet:
    """Apply :class:`~qiskit.circuit.library.CSGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
    from .library.standard_gates.s import CSGate
    return self.append(CSGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], [])