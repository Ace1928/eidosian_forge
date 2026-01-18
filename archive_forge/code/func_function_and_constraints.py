import scipy.sparse as sps
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from scipy.sparse.linalg import LinearOperator
def function_and_constraints(self, z):
    """Returns barrier function and constraints at given point.

        For z = [x, s], returns barrier function:
            function(z) = fun(x) - barrier_parameter*sum(log(s))
        and barrier constraints:
            constraints(z) = [   constr_eq(x)     ]
                             [ constr_ineq(x) + s ]

        """
    x = self.get_variables(z)
    s = self.get_slack(z)
    f = self.fun(x)
    c_eq, c_ineq = self.constr(x)
    return (self._compute_function(f, c_ineq, s), self._compute_constr(c_ineq, c_eq, s))