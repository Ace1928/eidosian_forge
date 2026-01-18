import numpy as np
from scipy.optimize._trustregion_exact import (
from scipy.linalg import (svd, get_lapack_funcs, det, qr, norm)
from numpy.testing import (assert_array_equal,
def test_for_random_entries(self):
    np.random.seed(1)
    n = 5
    for case in ('easy', 'hard', 'jac_equal_zero'):
        eig_limits = [(-20, -15), (-10, -5), (-10, 0), (-5, 5), (-10, 10), (0, 10), (5, 10), (15, 20)]
        for min_eig, max_eig in eig_limits:
            H, g = random_entry(n, min_eig, max_eig, case)
            trust_radius_list = [0.1, 0.3, 0.6, 0.8, 1, 1.2, 3.3, 5.5, 10]
            for trust_radius in trust_radius_list:
                subprob_ac = IterativeSubproblem(0, lambda x: 0, lambda x: g, lambda x: H, k_easy=1e-10, k_hard=1e-10)
                p_ac, hits_boundary_ac = subprob_ac.solve(trust_radius)
                J_ac = 1 / 2 * np.dot(p_ac, np.dot(H, p_ac)) + np.dot(g, p_ac)
                stop_criteria = [(0.1, 2), (0.5, 1.1), (0.9, 1.01)]
                for k_opt, k_trf in stop_criteria:
                    k_easy = min(k_trf - 1, 1 - np.sqrt(k_opt))
                    k_hard = 1 - k_opt
                    subprob = IterativeSubproblem(0, lambda x: 0, lambda x: g, lambda x: H, k_easy=k_easy, k_hard=k_hard)
                    p, hits_boundary = subprob.solve(trust_radius)
                    J = 1 / 2 * np.dot(p, np.dot(H, p)) + np.dot(g, p)
                    if hits_boundary:
                        assert_array_equal(np.abs(norm(p) - trust_radius) <= (k_trf - 1) * trust_radius, True)
                    else:
                        assert_equal(norm(p) <= trust_radius, True)
                    assert_equal(J <= k_opt * J_ac, True)