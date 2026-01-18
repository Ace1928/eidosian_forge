from __future__ import annotations
from typing import TYPE_CHECKING, List
from collections import defaultdict
from collections.abc import Mapping, Sequence, Iterable
from numbers import Number
from copy import deepcopy
import numpy as np
import rustworkx as rx
from qiskit._accelerate.sparse_pauli_op import unordered_unique, decompose_dense
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.parametertable import ParameterView
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.custom_iterator import CustomIterator
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic.pauli import BasePauli
from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
@staticmethod
def from_sparse_list(obj: Iterable[tuple[str, list[int], complex]], num_qubits: int, do_checks: bool=True, dtype: type=complex) -> SparsePauliOp:
    """Construct from a list of local Pauli strings and coefficients.

        Each list element is a 3-tuple of a local Pauli string, indices where to apply it,
        and a coefficient.

        For example, the 5-qubit Hamiltonian

        .. math::

            H = Z_1 X_4 + 2 Y_0 Y_3

        can be constructed as

        .. code-block:: python

            # via triples and local Paulis with indices
            op = SparsePauliOp.from_sparse_list([("ZX", [1, 4], 1), ("YY", [0, 3], 2)], num_qubits=5)

            # equals the following construction from "dense" Paulis
            op = SparsePauliOp.from_list([("XIIZI", 1), ("IYIIY", 2)])

        Args:
            obj (Iterable[tuple[str, list[int], complex]]): The list 3-tuples specifying the Paulis.
            num_qubits (int): The number of qubits of the operator.
            do_checks (bool): The flag of checking if the input indices are not duplicated
            (Default: True).
            dtype (type): The dtype of coeffs (Default: complex).

        Returns:
            SparsePauliOp: The SparsePauliOp representation of the Pauli terms.

        Raises:
            QiskitError: If the number of qubits is incompatible with the indices of the Pauli terms.
            QiskitError: If the designated qubit is already assigned.
        """
    obj = list(obj)
    size = len(obj)
    if size == 0:
        obj = [('I' * num_qubits, range(num_qubits), 0)]
        size = len(obj)
    coeffs = np.zeros(size, dtype=dtype)
    labels = np.zeros(size, dtype=f'<U{num_qubits}')
    for i, (paulis, indices, coeff) in enumerate(obj):
        if do_checks and len(indices) != len(set(indices)):
            raise QiskitError('Input indices are duplicated.')
        label = ['I'] * num_qubits
        for pauli, index in zip(paulis, indices):
            if index >= num_qubits:
                raise QiskitError(f'The number of qubits ({num_qubits}) is smaller than a required index {index}.')
            label[~index] = pauli
        labels[i] = ''.join(label)
        coeffs[i] = coeff
    paulis = PauliList(labels)
    return SparsePauliOp(paulis, coeffs, copy=False)