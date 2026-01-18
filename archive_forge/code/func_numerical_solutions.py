from .polynomial import Polynomial
from .component import NonZeroDimensionalComponent
from ..pari import pari
def numerical_solutions(polys):
    polysReduced = [poly.factor_out_variables() for poly in polys]
    polysFiltered = [poly for poly in polysReduced if not poly.is_constant() or poly.get_constant() == 0]
    for poly in polysFiltered:
        if poly.is_constant():
            return NonZeroDimensionalComponent()
    polysAndVars = [PariPolynomialAndVariables(poly) for poly in polysFiltered]
    solutions = _numerical_solutions_recursion(polysAndVars, {})
    number_variables = len(set(sum([poly.variables() for poly in polys], [])))
    return [solution if len(solution) == number_variables else NonZeroDimensionalComponent() for solution in solutions]