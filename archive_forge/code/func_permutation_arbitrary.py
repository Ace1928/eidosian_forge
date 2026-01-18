from typing import List, Sequence, Tuple, Union, cast
import numpy as np
from pyquil.experiment._setting import TensorProductState
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import Parameter
from pyquil.quilbase import Gate, Halt, _strip_modifiers
from pyquil.simulation.matrices import SWAP, STATES, QUANTUM_GATES
def permutation_arbitrary(qubit_inds: Sequence[int], n_qubits: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generate the permutation matrix that permutes an arbitrary number of
    single-particle Hilbert spaces into adjacent positions.

    Transposes the qubit indices in the order they are passed to a
    contiguous region in the complete Hilbert space, in increasing
    qubit index order (preserving the order they are passed in).

    Gates are usually defined as `GATE 0 1 2`, with such an argument ordering
    dictating the layout of the matrix corresponding to GATE. If such an
    instruction is given, actual qubits (0, 1, 2) need to be swapped into the
    positions (2, 1, 0), because the lifting operation taking the 8 x 8 matrix
    of GATE is done in the little-endian (reverse) addressed qubit space.

    For example, suppose I have a Quil command CCNOT 20 15 10.
    The median of the qubit indices is 15 - hence, we permute qubits
    [20, 15, 10] into the final map [16, 15, 14] to minimize the number of
    swaps needed, and so we can directly operate with the final CCNOT, when
    lifted from indices [16, 15, 14] to the complete Hilbert space.

    Notes: assumes qubit indices are unique (assured in parent call).

    See documentation for further details and explanation.

    Done in preparation for arbitrary gate application on
    adjacent qubits.

    :param qubit_inds: Qubit indices in the order the gate is applied to.
    :param n_qubits: Number of qubits in system
    :return:
        perm - permutation matrix providing the desired qubit reordering
        qubit_arr - new indexing of qubits presented in left to right decreasing index order.
        start_i - starting index to lift gate from
    """
    perm = np.eye(2 ** n_qubits, dtype=np.complex128)
    sorted_inds = np.sort(qubit_inds)
    med_i = len(qubit_inds) // 2
    med = sorted_inds[med_i]
    start = med - med_i
    final_map = np.arange(start, start + len(qubit_inds))[::-1]
    start_i = final_map[-1]
    qubit_arr = np.arange(n_qubits)
    made_it = False
    right = True
    while not made_it:
        array = range(len(qubit_inds)) if right else range(len(qubit_inds))[::-1]
        for i in array:
            pmod, qubit_arr = two_swap_helper(np.where(qubit_arr == qubit_inds[i])[0][0], final_map[i], n_qubits, qubit_arr)
            perm = pmod.dot(perm)
            if np.allclose(qubit_arr[final_map[-1]:final_map[0] + 1][::-1], qubit_inds):
                made_it = True
                break
        right = not right
    assert np.allclose(qubit_arr[final_map[-1]:final_map[0] + 1][::-1], qubit_inds)
    return (perm, qubit_arr[::-1], start_i)