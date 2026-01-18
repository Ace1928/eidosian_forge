import math
import cmath
import numpy as np
from scipy.special import factorial as fac
import pennylane as qml
from pennylane.ops import Identity
from pennylane import Device
from .._version import __version__
def homodyne(phi=None):
    """Function factory that returns the Homodyne expectation of a one mode state.

    Args:
        phi (float): the default phase space axis to perform the Homodyne measurement

    Returns:
        function: A function that accepts a single mode means vector, covariance matrix,
        and phase space angle phi, and returns the quadrature expectation
        value and variance.
    """
    if phi is not None:

        def _homodyne(cov, mu, params, hbar=2.0):
            """Arbitrary angle homodyne expectation."""
            rot = rotation(phi)
            muphi = rot.T @ mu
            covphi = rot.T @ cov @ rot
            return (muphi[0], covphi[0, 0])
        return _homodyne

    def _homodyne(cov, mu, params, hbar=2.0):
        """Arbitrary angle homodyne expectation."""
        rot = rotation(params[0])
        muphi = rot.T @ mu
        covphi = rot.T @ cov @ rot
        return (muphi[0], covphi[0, 0])
    return _homodyne