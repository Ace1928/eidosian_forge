from functools import wraps
from inspect import signature
import warnings
import numpy as np
from autoray import numpy as anp
import pennylane as qml
def constant_fn(x):
    """Univariate reconstruction of a constant Fourier series."""
    return f0