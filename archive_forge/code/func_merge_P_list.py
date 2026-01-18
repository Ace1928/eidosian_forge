from __future__ import annotations, division
import operator
from typing import List
import numpy as np
import scipy.sparse as sp
from cvxpy.cvxcore.python import canonInterface
from cvxpy.lin_ops.canon_backend import TensorRepresentation
from cvxpy.lin_ops.lin_op import NO_OP, LinOp
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.utilities.replace_quad_forms import (
def merge_P_list(self, P_list: List[TensorRepresentation], P_height: int, num_params: int) -> sp.csc_matrix:
    """Conceptually we build a block diagonal matrix
           out of all the Ps, then flatten the first two dimensions.
           eg P1
                P2
           We do this by extending each P with zero blocks above and below.

        Args:
            P_list: list of P submatrices as TensorRepresentation objects.
            P_entries: number of entries in the merged P matrix.
            P_height: number of rows in the merged P matrix.
            num_params: number of parameters in the problem.
        
        Returns:
            A CSC sparse representation of the merged P matrix.
        """
    offset = 0
    for P in P_list:
        m, n = P.shape
        assert m == n
        P.row += offset
        P.col += offset
        P.shape = (P_height, P_height)
        offset += m
    combined = TensorRepresentation.combine(P_list)
    return combined.flatten_tensor(num_params)