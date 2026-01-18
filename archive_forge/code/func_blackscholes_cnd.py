import math
import numpy as np
from numba import jit
def blackscholes_cnd(d):
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.3989422804014327
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = RSQRT2PI * math.exp(-0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val