from math import prod
from sympy.core.mul import Mul
from sympy.matrices.dense import (Matrix, diag)
from sympy.polys.polytools import (Poly, degree_list, rem)
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import IndexedBase
from sympy.polys.monomials import itermonomials, monomial_deg
from sympy.polys.orderings import monomial_key
from sympy.polys.polytools import poly_from_expr, total_degree
from sympy.functions.combinatorial.factorials import binomial
from itertools import combinations_with_replacement
from sympy.utilities.exceptions import sympy_deprecation_warning
def get_submatrix(self, matrix):
    """
        Returns
        =======

        macaulay_submatrix: Matrix
            The Macaulay denominator matrix. Columns that are non reduced are kept.
            The row which contains one of the a_{i}s is dropped. a_{i}s
            are the coefficients of x_i ^ {d_i}.
        """
    reduced, non_reduced = self.get_reduced_nonreduced()
    if reduced == []:
        return diag([1])
    reduction_set = [v ** self.degrees[i] for i, v in enumerate(self.variables)]
    ais = [self.polynomials[i].coeff(reduction_set[i]) for i in range(self.n)]
    reduced_matrix = matrix[:, reduced]
    keep = []
    for row in range(reduced_matrix.rows):
        check = [ai in reduced_matrix[row, :] for ai in ais]
        if True not in check:
            keep.append(row)
    return matrix[keep, non_reduced]