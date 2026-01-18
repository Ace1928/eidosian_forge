from sympy.core.backend import eye, Matrix, zeros
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.mechanics.functions import find_dynamicsymbols
@property
def comb_implicit_rhs(self):
    """Returns the column matrix, F, corresponding to the equations of
        motion in implicit form (form [2]), M x' = F, where the kinematical
        equations are included"""
    if self._comb_implicit_rhs is None:
        if self._dyn_implicit_rhs is not None:
            kin_inter = self._kin_explicit_rhs
            dyn_inter = self._dyn_implicit_rhs
            self._comb_implicit_rhs = kin_inter.col_join(dyn_inter)
            return self._comb_implicit_rhs
        else:
            raise AttributeError('comb_implicit_mat is not specified for equations of motion in form [1].')
    else:
        return self._comb_implicit_rhs