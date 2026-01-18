from .polynomial import Polynomial, Monomial
from . import matrix
def degree_of_monomial(m):
    vars = dict(m.get_vars())
    return vars.get(mod_var, 0)