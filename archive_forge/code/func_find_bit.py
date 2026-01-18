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
def find_bit(self, bit: Bit) -> BitLocations:
    """Find locations in the circuit which can be used to reference a given :obj:`~Bit`.

        Args:
            bit (Bit): The bit to locate.

        Returns:
            namedtuple(int, List[Tuple(Register, int)]): A 2-tuple. The first element (``index``)
                contains the index at which the ``Bit`` can be found (in either
                :obj:`~QuantumCircuit.qubits`, :obj:`~QuantumCircuit.clbits`, depending on its
                type). The second element (``registers``) is a list of ``(register, index)``
                pairs with an entry for each :obj:`~Register` in the circuit which contains the
                :obj:`~Bit` (and the index in the :obj:`~Register` at which it can be found).

        Notes:
            The circuit index of an :obj:`~AncillaQubit` will be its index in
            :obj:`~QuantumCircuit.qubits`, not :obj:`~QuantumCircuit.ancillas`.

        Raises:
            CircuitError: If the supplied :obj:`~Bit` was of an unknown type.
            CircuitError: If the supplied :obj:`~Bit` could not be found on the circuit.
        """
    try:
        if isinstance(bit, Qubit):
            return self._qubit_indices[bit]
        elif isinstance(bit, Clbit):
            return self._clbit_indices[bit]
        else:
            raise CircuitError(f'Could not locate bit of unknown type: {type(bit)}')
    except KeyError as err:
        raise CircuitError(f'Could not locate provided bit: {bit}. Has it been added to the QuantumCircuit?') from err