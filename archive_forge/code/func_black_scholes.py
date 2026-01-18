import numpy as np
import math
from numba import cuda, double, void
from numba.cuda.testing import unittest, CUDATestCase
def black_scholes(callResult, putResult, stockPrice, optionStrike, optionYears, Riskfree, Volatility):
    S = stockPrice
    X = optionStrike
    T = optionYears
    R = Riskfree
    V = Volatility
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd(d1)
    cndd2 = cnd(d2)
    expRT = np.exp(-R * T)
    callResult[:] = S * cndd1 - X * expRT * cndd2
    putResult[:] = X * expRT * (1.0 - cndd2) - S * (1.0 - cndd1)