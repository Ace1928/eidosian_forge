import numpy as np
import scipy.sparse as sps
@classmethod
def from_PreparedConstraint(cls, constraint):
    """Create an instance from `PreparedConstrained` object."""
    lb, ub = constraint.bounds
    cfun = constraint.fun
    keep_feasible = constraint.keep_feasible
    if np.all(lb == -np.inf) and np.all(ub == np.inf):
        return cls.empty(cfun.n)
    if np.all(lb == -np.inf) and np.all(ub == np.inf):
        return cls.empty(cfun.n)
    elif np.all(lb == ub):
        return cls._equal_to_canonical(cfun, lb)
    elif np.all(lb == -np.inf):
        return cls._less_to_canonical(cfun, ub, keep_feasible)
    elif np.all(ub == np.inf):
        return cls._greater_to_canonical(cfun, lb, keep_feasible)
    else:
        return cls._interval_to_canonical(cfun, lb, ub, keep_feasible)