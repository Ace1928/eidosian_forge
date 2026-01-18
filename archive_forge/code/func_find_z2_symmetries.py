from __future__ import annotations
import itertools
from collections.abc import Iterable
from copy import deepcopy
from typing import Union, cast
import numpy as np
from qiskit.exceptions import QiskitError
from ..operators import Pauli, SparsePauliOp
@classmethod
def find_z2_symmetries(cls, operator: SparsePauliOp) -> Z2Symmetries:
    """
        Finds Z2 Pauli-type symmetries of a :class:`.SparsePauliOp`.

        Returns:
            A ``Z2Symmetries`` instance.
        """
    pauli_symmetries = []
    sq_paulis = []
    sq_list = []
    stacked_paulis = []
    test_idx = {'X_or_I': [(0, 0), (1, 0)], 'Y_or_I': [(0, 0), (1, 1)], 'Z_or_I': [(0, 0), (0, 1)]}
    test_row = {'Z_or_I': [(1, 0), (1, 1)], 'X_or_I': [(0, 1), (1, 1)], 'Y_or_I': [(0, 1), (1, 0)]}
    pauli_bool = {'Z_or_I': [False, True], 'X_or_I': [True, False], 'Y_or_I': [True, True]}
    if _sparse_pauli_op_is_zero(operator):
        return cls([], [], [], None)
    for pauli in iter(operator):
        stacked_paulis.append(np.concatenate((pauli.paulis.x[0], pauli.paulis.z[0]), axis=0).astype(int))
    stacked_matrix = np.stack(stacked_paulis)
    symmetries = _kernel_f2(stacked_matrix)
    if not symmetries:
        return cls([], [], [], None)
    stacked_symmetries = np.stack(symmetries)
    symm_shape = stacked_symmetries.shape
    half_symm_shape = symm_shape[1] // 2
    stacked_symm_del = [np.delete(stacked_symmetries, row, axis=0) for row in range(symm_shape[0])]

    def _test_symmetry_row_col(row: int, col: int, idx_test: list, row_test: list) -> bool:
        """
            Utility method that determines how to build the list of single-qubit Pauli X operators and
            the list of corresponding qubit indices from the stacked symmetries.
            This method is successively applied to Z type, X type and Y type symmetries (in this order)
            to build the letter at position (col) of the Pauli word corresponding to the symmetry at
            position (row).

            Args:
                row (int): Index of the symmetry for which the single-qubit Pauli X operator is being
                    built.
                col (int): Index of the letter in the Pauli word corresponding to the single-qubit Pauli
                    X operator.
                idx_test (list): List of possibilities for the stacked symmetries at all other rows
                    than row.
                row_test (list): List of possibilities for the stacked symmetries at row.

            Returns:
                Whether or not this symmetry type should be used to build this letter of this
                single-qubit Pauli X operator.
            """
        stacked_symm_idx_tests = np.array([(stacked_symm_del[row][symm_idx, col], stacked_symm_del[row][symm_idx, col + half_symm_shape]) in idx_test for symm_idx in range(symm_shape[0] - 1)])
        stacked_symm_row_test = (stacked_symmetries[row, col], stacked_symmetries[row, col + half_symm_shape]) in row_test
        return bool(np.all(stacked_symm_idx_tests)) and stacked_symm_row_test
    for row in range(symm_shape[0]):
        pauli_symmetries.append(Pauli((stacked_symmetries[row, :half_symm_shape], stacked_symmetries[row, half_symm_shape:])))
        for col in range(half_symm_shape):
            for key in ('Z_or_I', 'X_or_I', 'Y_or_I'):
                current_test_result = _test_symmetry_row_col(row, col, test_idx[key], test_row[key])
                if current_test_result:
                    sq_paulis.append(Pauli((np.zeros(half_symm_shape), np.zeros(half_symm_shape))))
                    sq_paulis[row].z[col] = pauli_bool[key][0]
                    sq_paulis[row].x[col] = pauli_bool[key][1]
                    sq_list.append(col)
                    break
            if current_test_result:
                break
    return cls(pauli_symmetries, sq_paulis, sq_list, None)