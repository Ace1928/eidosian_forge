from sympy.core.backend import eye, Matrix, zeros
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.mechanics.functions import find_dynamicsymbols
@property
def alg_con(self):
    """Returns a list with the indices of the rows containing algebraic
        constraints in the combined form of the equations of motion"""
    return self._alg_con