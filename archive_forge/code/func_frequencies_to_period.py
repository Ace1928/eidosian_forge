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
def frequencies_to_period(frequencies, decimals=5):
    """Returns the period of a Fourier series as defined
    by a set of frequencies.

    The period is simply :math:`2\\pi/gcd(frequencies)`,
    where :math:`\\text{gcd}` is the greatest common divisor.

    Args:
        spectra (tuple[int, float]): frequency spectra
        decimals (int): Number of decimal places to round to
            if there are non-integral frequencies.

    Returns:
        tuple[int, float]: frequencies

    **Example**

    >>> frequencies = (0.5, 1.0)
    >>> frequencies_to_period(frequencies)
    12.566370614359172
    """
    try:
        gcd = np.gcd.reduce(frequencies)
    except TypeError:
        exponent = 10 ** decimals
        frequencies = np.round(frequencies, decimals) * exponent
        gcd = np.gcd.reduce(np.int64(frequencies)) / exponent
    return 2 * np.pi / gcd