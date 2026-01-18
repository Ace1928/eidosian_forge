from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, curve_fit
from time import time
def arclength(eps, a, b, x, epsrel=0.01, limit=100):
    """Compute Arc length of f.

        Note that the arc length of a function f from t0 to t1 is given by
            int_t0^t1 sqrt(1 + f'(t)^2) dt
        """
    return quad(lambda phi: np.sqrt(1 + fp(eps, a, b, x, phi) ** 2), 0, np.pi, epsrel=epsrel, limit=100)[0]