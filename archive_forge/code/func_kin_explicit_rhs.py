from sympy.core.backend import eye, Matrix, zeros
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.mechanics.functions import find_dynamicsymbols
@property
def kin_explicit_rhs(self):
    """Returns the right hand side of the kinematical equations in explicit
        form, q' = G"""
    if self._kin_explicit_rhs is None:
        raise AttributeError('kin_explicit_rhs is not specified for equations of motion form [1] or [2].')
    else:
        return self._kin_explicit_rhs