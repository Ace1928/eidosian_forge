import math
import warnings
from itertools import combinations_with_replacement
import cupy as cp
def polynomial_matrix(x, powers, out):
    """Evaluate monomials, with exponents from `powers`, at `x`."""
    pwr = x[:, None, :] ** powers[None, :, :]
    cp.prod(pwr, axis=-1, out=out)