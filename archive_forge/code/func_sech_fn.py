from __future__ import annotations
import functools
import numpy as np
from qiskit.pulse.exceptions import PulseError
def sech_fn(x, *args, **kwargs):
    """Hyperbolic secant function"""
    return 1.0 / np.cosh(x, *args, **kwargs)