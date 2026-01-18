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
def prepare_state(self, state: Statevector | Sequence[complex] | str | int, qubits: Sequence[QubitSpecifier] | None=None, label: str | None=None, normalize: bool=False) -> InstructionSet:
    """Prepare qubits in a specific state.

        This class implements a state preparing unitary. Unlike
        :meth:`.initialize` it does not reset the qubits first.

        Args:
            state: The state to initialize to, can be either of the following.

                * Statevector or vector of complex amplitudes to initialize to.
                * Labels of basis states of the Pauli eigenstates Z, X, Y. See
                  :meth:`.Statevector.from_label`. Notice the order of the labels is reversed with
                  respect to the qubit index to be applied to. Example label '01' initializes the
                  qubit zero to :math:`|1\\rangle` and the qubit one to :math:`|0\\rangle`.
                * An integer that is used as a bitmap indicating which qubits to initialize to
                  :math:`|1\\rangle`. Example: setting params to 5 would initialize qubit 0 and qubit
                  2 to :math:`|1\\rangle` and qubit 1 to :math:`|0\\rangle`.

            qubits: Qubits to initialize. If ``None`` the initialization is applied to all qubits in
                the circuit.
            label: An optional label for the gate
            normalize: Whether to normalize an input array to a unit vector.

        Returns:
            A handle to the instruction that was just initialized

        Examples:
            Prepare a qubit in the state :math:`(|0\\rangle - |1\\rangle) / \\sqrt{2}`.

            .. code-block::

                import numpy as np
                from qiskit import QuantumCircuit

                circuit = QuantumCircuit(1)
                circuit.prepare_state([1/np.sqrt(2), -1/np.sqrt(2)], 0)
                circuit.draw()

            output:

            .. parsed-literal::

                     ┌─────────────────────────────────────┐
                q_0: ┤ State Preparation(0.70711,-0.70711) ├
                     └─────────────────────────────────────┘


            Prepare from a string two qubits in the state :math:`|10\\rangle`.
            The order of the labels is reversed with respect to qubit index.
            More information about labels for basis states are in
            :meth:`.Statevector.from_label`.

            .. code-block::

                import numpy as np
                from qiskit import QuantumCircuit

                circuit = QuantumCircuit(2)
                circuit.prepare_state('01', circuit.qubits)
                circuit.draw()

            output:

            .. parsed-literal::

                     ┌─────────────────────────┐
                q_0: ┤0                        ├
                     │  State Preparation(0,1) │
                q_1: ┤1                        ├
                     └─────────────────────────┘


            Initialize two qubits from an array of complex amplitudes
            .. code-block::

                import numpy as np
                from qiskit import QuantumCircuit

                circuit = QuantumCircuit(2)
                circuit.prepare_state([0, 1/np.sqrt(2), -1.j/np.sqrt(2), 0], circuit.qubits)
                circuit.draw()

            output:

            .. parsed-literal::

                     ┌───────────────────────────────────────────┐
                q_0: ┤0                                          ├
                     │  State Preparation(0,0.70711,-0.70711j,0) │
                q_1: ┤1                                          ├
                     └───────────────────────────────────────────┘
        """
    from qiskit.circuit.library.data_preparation import StatePreparation
    if qubits is None:
        qubits = self.qubits
    elif isinstance(qubits, (int, np.integer, slice, Qubit)):
        qubits = [qubits]
    num_qubits = len(qubits) if isinstance(state, int) else None
    return self.append(StatePreparation(state, num_qubits, label=label, normalize=normalize), qubits)