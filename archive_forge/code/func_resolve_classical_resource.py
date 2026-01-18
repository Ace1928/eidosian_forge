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
def resolve_classical_resource(self, specifier):
    if self._built:
        raise CircuitError('Cannot add resources after the scope has been built.')
    resource = self._parent.resolve_classical_resource(specifier)
    if isinstance(resource, Clbit):
        self.add_bits((resource,))
    else:
        self.add_register(resource)
    return resource