from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, curve_fit
from time import time
def pg_series(k, z, n):
    """Symbolic expansion of polygamma(k, z) in z=0 to order n."""
    return sympy.diff(dg_series(z, n + k), z, k)