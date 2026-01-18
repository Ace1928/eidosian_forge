from __future__ import annotations
import abc
import itertools
import typing
from typing import Collection, Iterable, FrozenSet, Tuple, Union, Optional, Sequence
from qiskit._accelerate.quantum_circuit import CircuitData
from qiskit.circuit.classicalregister import Clbit, ClassicalRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuitdata import CircuitInstruction
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.circuit.register import Register
from ._builder_utils import condition_resources, node_resources
@abc.abstractmethod
def concrete_instruction(self, qubits: FrozenSet[Qubit], clbits: FrozenSet[Clbit]) -> Tuple[Instruction, InstructionResources]:
    """Get a concrete, complete instruction that is valid to act over all the given resources.

        The returned resources may not be the full width of the given resources, but will certainly
        be a subset of them; this can occur if (for example) a placeholder ``if`` statement is
        present, but does not itself contain any placeholder instructions.  For resource efficiency,
        the returned :class:`.ControlFlowOp` will not unnecessarily span all resources, but only the
        ones that it needs.

        .. note::

            The caller of this function is responsible for ensuring that the inputs to this function
            are non-strict supersets of the bits returned by :meth:`placeholder_resources`.

        Any condition added in by a call to :obj:`.Instruction.c_if` will be propagated through, but
        set properties like ``duration`` will not; it doesn't make sense for control-flow operations
        to have pulse scheduling on them.

        Args:
            qubits: The qubits the created instruction should be defined across.
            clbits: The clbits the created instruction should be defined across.

        Returns:
            A full version of the relevant control-flow instruction, and the resources that it uses.
            This is a "proper" instruction instance, as if it had been defined with the correct
            number of qubits and clbits from the beginning.
        """
    raise NotImplementedError