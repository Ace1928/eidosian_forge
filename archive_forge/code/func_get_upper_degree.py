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
def get_upper_degree(self):
    sympy_deprecation_warning('\n            The get_upper_degree() method of DixonResultant is deprecated. Use\n            get_max_degrees() instead.\n            ', deprecated_since_version='1.5', active_deprecations_target='deprecated-dixonresultant-properties')
    list_of_products = [self.variables[i] ** self._max_degrees[i] for i in range(self.n)]
    product = prod(list_of_products)
    product = Poly(product).monoms()
    return monomial_deg(*product)