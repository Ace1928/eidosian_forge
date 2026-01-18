import logging
from typing import Optional, List, Tuple, Dict
import numpy as np
from qiskit.exceptions import QiskitError
from ..utils import marginal_counts
from ..counts import Counts
def expval_with_stddev(coeffs: np.ndarray, probs: np.ndarray, shots: int) -> Tuple[float, float]:
    """Compute expectation value and standard deviation.
    Args:
        coeffs: array of diagonal operator coefficients.
        probs: array of measurement probabilities.
        shots: total number of shots to obtain probabilities.
    Returns:
        tuple: (expval, stddev) expectation value and standard deviation.
    """
    expval = coeffs.dot(probs)
    sq_expval = (coeffs ** 2).dot(probs)
    variance = (sq_expval - expval ** 2) / shots
    if variance < 0 and (not np.isclose(variance, 0)):
        logger.warning('Encountered a negative variance in expectation value calculation.(%f). Setting standard deviation of result to 0.', variance)
    calc_stddev = np.sqrt(variance) if variance > 0 else 0.0
    return [expval, calc_stddev]