from sympy.core.backend import eye, Matrix, zeros
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.mechanics.functions import find_dynamicsymbols
def compute_explicit_form(self):
    """If the explicit right hand side of the combined equations of motion
        is to provided upon initialization, this method will calculate it. This
        calculation can potentially take awhile to compute."""
    if self._comb_explicit_rhs is not None:
        raise AttributeError('comb_explicit_rhs is already formed.')
    inter1 = getattr(self, 'kin_explicit_rhs', None)
    if inter1 is not None:
        inter2 = self._dyn_implicit_mat.LUsolve(self._dyn_implicit_rhs)
        out = inter1.col_join(inter2)
    else:
        out = self._comb_implicit_mat.LUsolve(self._comb_implicit_rhs)
    self._comb_explicit_rhs = out