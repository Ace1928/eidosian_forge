import unittest
import numpy as np
from statsmodels.multivariate.factor_rotation._wrappers import rotate_factors
from statsmodels.multivariate.factor_rotation._gpa_rotation import (
from statsmodels.multivariate.factor_rotation._analytic_rotation import (
def test_equivalence_orthomax_oblimin(self):
    """
        These criteria should be equivalent when restricted to orthogonal
        rotation.
        See Hartman 1976 page 299.
        """
    A = self.get_A()
    gamma = 0
    vgQ = lambda L=None, A=None, T=None: orthomax_objective(L=L, A=A, T=T, gamma=gamma, return_gradient=True)
    L_orthomax, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
    vgQ = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=gamma, rotation_method='orthogonal', return_gradient=True)
    L_oblimin, phi2, T2, table2 = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
    self.assertTrue(np.allclose(L_orthomax, L_oblimin, atol=1e-05))
    gamma = 1
    vgQ = lambda L=None, A=None, T=None: orthomax_objective(L=L, A=A, T=T, gamma=gamma, return_gradient=True)
    L_orthomax, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
    vgQ = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=gamma, rotation_method='orthogonal', return_gradient=True)
    L_oblimin, phi2, T2, table2 = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
    self.assertTrue(np.allclose(L_orthomax, L_oblimin, atol=1e-05))