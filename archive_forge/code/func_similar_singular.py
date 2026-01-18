from typing import Tuple, Callable, List
import numpy as np
from cirq.linalg import combinators, predicates, tolerance
def similar_singular(i, j):
    return np.allclose(diagonal_matrix[i, i], diagonal_matrix[j, j], rtol=rtol)