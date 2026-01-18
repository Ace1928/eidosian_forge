import itertools
from functools import reduce
from typing import Generator, Iterable, Tuple
import numpy as np
from scipy.sparse import csr_matrix, eye, kron
import pennylane as qml
from pennylane.wires import Wires
def reduce_matrices(mats_and_wires_gen: Generator[Tuple[np.ndarray, Wires], None, None], reduce_func: callable) -> Tuple[np.ndarray, Wires]:
    """Apply the given ``reduce_func`` cumulatively to the items of the ``mats_and_wires_gen``
    generator, from left to right, so as to reduce the sequence to a tuple containing a single
    matrix and the wires it acts on.

    Args:
        mats_and_wires_gen (Generator): generator of tuples containing the matrix and the wires of
            each operator
        reduce_func (callable): function used to reduce the sequence of operators

    Returns:
        Tuple[tensor, Wires]: a tuple containing the reduced matrix and the wires it acts on
    """

    def expand_and_reduce(op1_tuple: Tuple[np.ndarray, Wires], op2_tuple: Tuple[np.ndarray, Wires]):
        mat1, wires1 = op1_tuple
        mat2, wires2 = op2_tuple
        expanded_wires = wires1 + wires2
        mat1 = expand_matrix(mat1, wires1, wire_order=expanded_wires)
        mat2 = expand_matrix(mat2, wires2, wire_order=expanded_wires)
        return (reduce_func(mat1, mat2), expanded_wires)
    reduced_mat, final_wires = reduce(expand_and_reduce, mats_and_wires_gen)
    return (reduced_mat, final_wires)