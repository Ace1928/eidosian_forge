from sympy.core.backend import eye, Matrix, zeros
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.mechanics.functions import find_dynamicsymbols
@property
def dyn_implicit_rhs(self):
    """Returns the column matrix, F, corresponding to the dynamic equations
        in implicit form, M x' = F, where the kinematical equations are not
        included"""
    if self._dyn_implicit_rhs is None:
        raise AttributeError('dyn_implicit_rhs is not specified for equations of motion form [1] or [2].')
    else:
        return self._dyn_implicit_rhs