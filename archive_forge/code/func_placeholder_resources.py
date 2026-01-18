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
def placeholder_resources(self) -> InstructionResources:
    """Get the qubit and clbit resources that this placeholder instruction should be considered
        as using before construction.

        This will likely not include *all* resources after the block has been built, but using the
        output of this method ensures that all resources will pass through a
        :meth:`.QuantumCircuit.append` call, even if they come from a placeholder, and consequently
        will be tracked by the scope managers.

        Returns:
            A collection of the quantum and classical resources this placeholder instruction will
            certainly use.
        """
    raise NotImplementedError