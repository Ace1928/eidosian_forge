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
def continue_loop(self) -> InstructionSet:
    """Apply :class:`~qiskit.circuit.ContinueLoopOp`.

        .. warning::

            If you are using the context-manager "builder" forms of :meth:`.if_test`,
            :meth:`.for_loop` or :meth:`.while_loop`, you can only call this method if you are
            within a loop context, because otherwise the "resource width" of the operation cannot be
            determined.  This would quickly lead to invalid circuits, and so if you are trying to
            construct a reusable loop body (without the context managers), you must also use the
            non-context-manager form of :meth:`.if_test` and :meth:`.if_else`.  Take care that the
            :class:`~qiskit.circuit.ContinueLoopOp` instruction must span all the resources of its
            containing loop, not just the immediate scope.

        Returns:
            A handle to the instruction created.

        Raises:
            CircuitError: if this method was called within a builder context, but not contained
                within a loop.
        """
    if self._control_flow_scopes:
        operation = ContinueLoopPlaceholder()
        resources = operation.placeholder_resources()
        return self.append(operation, resources.qubits, resources.clbits)
    return self.append(ContinueLoopOp(self.num_qubits, self.num_clbits), self.qubits, self.clbits)