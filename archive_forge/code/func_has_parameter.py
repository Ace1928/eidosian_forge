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
def has_parameter(self, name_or_param: str | Parameter, /) -> bool:
    """Check whether a parameter object exists in this circuit.

        Args:
            name_or_param: the parameter, or name of a parameter to check.  If this is a
                :class:`.Parameter` node, the parameter must be exactly the given one for this
                function to return ``True``.

        Returns:
            whether a matching parameter is assignable in this circuit.

        See also:
            :meth:`QuantumCircuit.get_parameter`
                Retrieve the :class:`.Parameter` instance from this circuit by name.
        """
    if isinstance(name_or_param, str):
        return self.get_parameter(name_or_param, None) is not None
    return self.get_parameter(name_or_param.name) == name_or_param