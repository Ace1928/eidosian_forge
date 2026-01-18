from typing import Any, Dict
import numpy as np
import cvxpy.interface as intf
def stack_vals(variables: list, default: float, order: str='F') -> np.ndarray:
    """Stacks the values of the given variables.

    Parameters
    ----------
    variables: list of cvxpy variables.
    default: value to use when variable value is None.
    order: unravel values in C or Fortran ("F") order

    Returns
    -------
    An initial guess for the solution.
    """
    value = []
    for variable in variables:
        if variable.value is not None:
            value.append(np.ravel(variable.value, order))
        else:
            value.append(np.full(variable.size, default))
    return np.concatenate(value)