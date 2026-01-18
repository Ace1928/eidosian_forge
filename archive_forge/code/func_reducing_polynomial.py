from .polynomial import Polynomial, Monomial
from . import matrix
def reducing_polynomial(m):

    def new_degree(var, expo):
        if var == mod_var:
            return (var, expo - mod_degree)
        else:
            return (var, expo)
    new_degrees = [new_degree(var, expo) for var, expo in m.get_vars()]
    new_degrees_filtered = tuple([(var, expo) for var, expo in new_degrees if expo > 0])
    monomial = Monomial(m.get_coefficient(), new_degrees_filtered)
    return Polynomial((monomial,))