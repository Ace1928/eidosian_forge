from .polynomial import Polynomial
from .fieldExtensions import my_rnfequation
from ..pari import pari
def extensions_and_assignments(polys):
    """
    Splits into extensions and assignments s in example given above
    in _exact_solutions.
    """
    extensions = []
    extension_vars = []
    assignments = {}
    while polys:
        poly, var = _next_var_and_poly(polys, extension_vars)
        polys = _remove(polys, poly)
        degree = poly.degree(var)
        if degree == 1:
            value = Polynomial.from_variable_name(var) - poly
            assert var not in value.variables()
            assert var not in assignments
            assignments[var] = value
        else:
            extensions.append((poly, var, degree))
            extension_vars.append(var)
    return (extensions, assignments)