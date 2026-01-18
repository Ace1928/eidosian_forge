import numpy as np
import matplotlib.pyplot as plt
import mpmath
def err_metric(a, b, atol=1e-290):
    m = abs(a - b) / (atol + abs(b))
    m[np.isinf(b) & (a == b)] = 0
    return m