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
def matrix_iter(self, sparse: bool=False):
    """Return a matrix representation iterator.

        This is a lazy iterator that converts each term in the SparsePauliOp
        into a matrix as it is used. To convert to a single matrix use the
        :meth:`to_matrix` method.

        Args:
            sparse (bool): optionally return sparse CSR matrices if True,
                           otherwise return Numpy array matrices
                           (Default: False)

        Returns:
            MatrixIterator: matrix iterator object for the PauliList.
        """

    class MatrixIterator(CustomIterator):
        """Matrix representation iteration and item access."""

        def __repr__(self):
            return f'<SparsePauliOp_matrix_iterator at {hex(id(self))}>'

        def __getitem__(self, key):
            coeff = self.obj.coeffs[key]
            mat = self.obj.paulis[key].to_matrix(sparse)
            return coeff * mat
    return MatrixIterator(self)