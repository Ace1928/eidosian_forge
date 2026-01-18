from typing import Optional, TYPE_CHECKING
import numpy as np
from cirq import linalg, value
def assert_allclose_up_to_global_phase(actual: np.ndarray, desired: np.ndarray, *, rtol: float=1e-07, atol: float, equal_nan: bool=True, err_msg: str='', verbose: bool=True) -> None:
    """Checks if a ~= b * exp(i t) for some t.

    Args:
        actual: A numpy array.
        desired: Another numpy array.
        rtol: Relative error tolerance.
        atol: Absolute error tolerance.
        equal_nan: Whether or not NaN entries should be considered equal to
            other NaN entries.
        err_msg: The error message to be printed in case of failure.
        verbose: If True, the conflicting values are appended to the error
            message.

    Raises:
        AssertionError: The matrices aren't nearly equal up to global phase.
    """
    __tracebackhide__ = True
    actual, desired = linalg.match_global_phase(actual, desired)
    np.testing.assert_allclose(actual=actual, desired=desired, rtol=rtol, atol=atol, equal_nan=equal_nan, err_msg=err_msg, verbose=verbose)