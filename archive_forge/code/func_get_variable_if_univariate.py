from .polynomial import Polynomial
from .component import NonZeroDimensionalComponent
from ..pari import pari
def get_variable_if_univariate(self):
    if len(self.variables) == 1:
        return self.variables[0]