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
def if_test(self, condition, true_body=None, qubits=None, clbits=None, *, label=None):
    """Create an ``if`` statement on this circuit.

        There are two forms for calling this function.  If called with all its arguments (with the
        possible exception of ``label``), it will create a
        :obj:`~qiskit.circuit.IfElseOp` with the given ``true_body``, and there will be
        no branch for the ``false`` condition (see also the :meth:`.if_else` method).  However, if
        ``true_body`` (and ``qubits`` and ``clbits``) are *not* passed, then this acts as a context
        manager, which can be used to build ``if`` statements.  The return value of the ``with``
        statement is a chainable context manager, which can be used to create subsequent ``else``
        blocks.  In this form, you do not need to keep track of the qubits or clbits you are using,
        because the scope will handle it for you.

        For example::

            from qiskit.circuit import QuantumCircuit, Qubit, Clbit
            bits = [Qubit(), Qubit(), Qubit(), Clbit(), Clbit()]
            qc = QuantumCircuit(bits)

            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 1)

            with qc.if_test((bits[3], 0)) as else_:
                qc.x(2)
            with else_:
                qc.h(2)
                qc.z(2)

        Args:
            condition (Tuple[Union[ClassicalRegister, Clbit], int]): A condition to be evaluated at
                circuit runtime which, if true, will trigger the evaluation of ``true_body``. Can be
                specified as either a tuple of a ``ClassicalRegister`` to be tested for equality
                with a given ``int``, or as a tuple of a ``Clbit`` to be compared to either a
                ``bool`` or an ``int``.
            true_body (Optional[QuantumCircuit]): The circuit body to be run if ``condition`` is
                true.
            qubits (Optional[Sequence[QubitSpecifier]]): The circuit qubits over which the if/else
                should be run.
            clbits (Optional[Sequence[ClbitSpecifier]]): The circuit clbits over which the if/else
                should be run.
            label (Optional[str]): The string label of the instruction in the circuit.

        Returns:
            InstructionSet or IfContext: depending on the call signature, either a context
            manager for creating the ``if`` block (it will automatically be added to the circuit at
            the end of the block), or an :obj:`~InstructionSet` handle to the appended conditional
            operation.

        Raises:
            CircuitError: If the provided condition references Clbits outside the
                enclosing circuit.
            CircuitError: if an incorrect calling convention is used.

        Returns:
            A handle to the instruction created.
        """
    circuit_scope = self._current_scope()
    if isinstance(condition, expr.Expr):
        condition = _validate_expr(circuit_scope, condition)
    else:
        condition = (circuit_scope.resolve_classical_resource(condition[0]), condition[1])
    if true_body is None:
        if qubits is not None or clbits is not None:
            raise CircuitError("When using 'if_test' as a context manager, you cannot pass qubits or clbits.")
        in_loop = bool(self._control_flow_scopes and self._control_flow_scopes[-1].allow_jumps)
        return IfContext(self, condition, in_loop=in_loop, label=label)
    elif qubits is None or clbits is None:
        raise CircuitError("When using 'if_test' with a body, you must pass qubits and clbits.")
    return self.append(IfElseOp(condition, true_body, None, label), qubits, clbits)