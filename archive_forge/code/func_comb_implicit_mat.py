from sympy.core.backend import eye, Matrix, zeros
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.mechanics.functions import find_dynamicsymbols
@property
def comb_implicit_mat(self):
    """Returns the matrix, M, corresponding to the equations of motion in
        implicit form (form [2]), M x' = F, where the kinematical equations are
        included"""
    if self._comb_implicit_mat is None:
        if self._dyn_implicit_mat is not None:
            num_kin_eqns = len(self._kin_explicit_rhs)
            num_dyn_eqns = len(self._dyn_implicit_rhs)
            zeros1 = zeros(num_kin_eqns, num_dyn_eqns)
            zeros2 = zeros(num_dyn_eqns, num_kin_eqns)
            inter1 = eye(num_kin_eqns).row_join(zeros1)
            inter2 = zeros2.row_join(self._dyn_implicit_mat)
            self._comb_implicit_mat = inter1.col_join(inter2)
            return self._comb_implicit_mat
        else:
            raise AttributeError('comb_implicit_mat is not specified for equations of motion form [1].')
    else:
        return self._comb_implicit_mat