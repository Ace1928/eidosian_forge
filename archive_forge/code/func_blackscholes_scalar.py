import math
import numpy as np
import unittest
from numba import njit
from numba.extending import register_jitable
from numba.tests.support import TestCase
@njit
def blackscholes_scalar(callResult, putResult, stockPrice, optionStrike, optionYears, Riskfree, Volatility):
    S = stockPrice
    X = optionStrike
    T = optionYears
    R = Riskfree
    V = Volatility
    for i in range(len(S)):
        sqrtT = math.sqrt(T[i])
        d1 = (math.log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
        d2 = d1 - V * sqrtT
        cndd1 = cnd(d1)
        cndd2 = cnd(d2)
        expRT = math.exp(-1.0 * R * T[i])
        callResult[i] = S[i] * cndd1 - X[i] * expRT * cndd2
        putResult[i] = X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1)