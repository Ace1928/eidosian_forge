import functools
import itertools
import numbers
import warnings
import numpy as np
from scipy.linalg import solve as linalg_solve
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.ops.functions import bind_new_parameters
from pennylane.tape import QuantumScript
@functools.lru_cache(maxsize=None)
def eigvals_to_frequencies(eigvals):
    """Convert an eigenvalue spectrum to frequency values, defined
    as the the set of positive, unique differences of the eigenvalues in the spectrum.

    Args:
        eigvals (tuple[int, float]): eigenvalue spectra

    Returns:
        tuple[int, float]: frequencies

    **Example**

    >>> eigvals = (-0.5, 0, 0, 0.5)
    >>> eigvals_to_frequencies(eigvals)
    (0.5, 1.0)
    """
    unique_eigvals = sorted(set(eigvals))
    return tuple({j - i for i, j in itertools.combinations(unique_eigvals, 2)})