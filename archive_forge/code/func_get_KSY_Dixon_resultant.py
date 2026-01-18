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
def get_KSY_Dixon_resultant(self, matrix):
    """Calculate the Kapur-Saxena-Yang approach to the Dixon Resultant."""
    matrix = self.delete_zero_rows_and_columns(matrix)
    _, U, _ = matrix.LUdecomposition()
    matrix = self.delete_zero_rows_and_columns(simplify(U))
    return self.product_leading_entries(matrix)