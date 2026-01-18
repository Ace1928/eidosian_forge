import scipy.sparse as sps
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from scipy.sparse.linalg import LinearOperator
def stop_criteria(self, state, z, last_iteration_failed, optimality, constr_violation, trust_radius, penalty, cg_info):
    """Stop criteria to the barrier problem.
        The criteria here proposed is similar to formula (2.3)
        from [1]_, p.879.
        """
    x = self.get_variables(z)
    if self.global_stop_criteria(state, x, last_iteration_failed, trust_radius, penalty, cg_info, self.barrier_parameter, self.tolerance):
        self.terminate = True
        return True
    else:
        g_cond = optimality < self.tolerance and constr_violation < self.tolerance
        x_cond = trust_radius < self.xtol
        return g_cond or x_cond