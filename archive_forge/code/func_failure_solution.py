import numpy as np
import cvxpy.settings as s
def failure_solution(status, attr=None) -> 'Solution':
    """Factory function for infeasible or unbounded solutions.

    Parameters
    ----------
    status : str
        The problem status.

    Returns
    -------
    Solution
        A solution object.
    """
    if status in [s.INFEASIBLE, s.INFEASIBLE_INACCURATE]:
        opt_val = np.inf
    elif status in [s.UNBOUNDED, s.UNBOUNDED_INACCURATE]:
        opt_val = -np.inf
    else:
        opt_val = None
    if attr is None:
        attr = {}
    if status == s.INFEASIBLE_OR_UNBOUNDED:
        attr['message'] = INF_OR_UNB_MESSAGE
    return Solution(status, opt_val, {}, {}, attr)