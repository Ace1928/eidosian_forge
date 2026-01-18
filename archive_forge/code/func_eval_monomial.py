from .polynomial import Polynomial
from .fieldExtensions import my_rnfequation
from ..pari import pari
def eval_monomial(monomial):
    return pari(monomial.get_coefficient()) * monomial_to_value[monomial.get_vars()]