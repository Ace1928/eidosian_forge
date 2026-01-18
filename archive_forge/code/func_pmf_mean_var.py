import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
@np.vectorize
def pmf_mean_var(x, N, m1, n, w):
    m2 = N - m1
    xl = np.maximum(0, n - m2)
    xu = np.minimum(n, m1)

    def f(x):
        t1 = special_binom(m1, x)
        t2 = special_binom(m2, n - x)
        return t1 * t2 * w ** x

    def P(k):
        return sum((f(y) * y ** k for y in range(xl, xu + 1)))
    P0 = P(0)
    P1 = P(1)
    P2 = P(2)
    pmf = f(x) / P0
    mean = P1 / P0
    var = P2 / P0 - (P1 / P0) ** 2
    return (pmf, mean, var)