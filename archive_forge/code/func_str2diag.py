import logging
from typing import Optional, List, Tuple, Dict
import numpy as np
from qiskit.exceptions import QiskitError
from ..utils import marginal_counts
from ..counts import Counts
def str2diag(string):
    """Transform diagonal from a string to a numpy array"""
    chars = {'I': np.array([1, 1], dtype=float), 'Z': np.array([1, -1], dtype=float), '0': np.array([1, 0], dtype=float), '1': np.array([0, 1], dtype=float)}
    ret = np.array([1], dtype=float)
    for i in reversed(string):
        if i not in chars:
            raise QiskitError(f'Invalid diagonal string character {i}')
        ret = np.kron(chars[i], ret)
    return ret