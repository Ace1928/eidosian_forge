import numpy as np
from scipy import optimize
from pygsp import utils
from . import Filter  # prevent circular import in Python < 3.5
def kernel_abspline3(x, alpha, beta, t1, t2):
    M = np.array([[1, t1, t1 ** 2, t1 ** 3], [1, t2, t2 ** 2, t2 ** 3], [0, 1, 2 * t1, 3 * t1 ** 2], [0, 1, 2 * t2, 3 * t2 ** 2]])
    v = np.array([1, 1, t1 ** (-alpha) * alpha * t1 ** (alpha - 1), -beta * t2 ** (-beta - 1) * t2 ** beta])
    a = np.linalg.solve(M, v)
    r1 = x <= t1
    r2 = (x >= t1) * (x < t2)
    r3 = x >= t2
    if isinstance(x, np.float64):
        if r1:
            r = x[r1] ** alpha * t1 ** (-alpha)
        if r2:
            r = a[0] + a[1] * x + a[2] * x ** 2 + a[3] * x ** 3
        if r3:
            r = x[r3] ** (-beta) * t2 ** beta
    else:
        r = np.zeros(x.shape)
        x2 = x[r2]
        r[r1] = x[r1] ** alpha * t1 ** (-alpha)
        r[r2] = a[0] + a[1] * x2 + a[2] * x2 ** 2 + a[3] * x2 ** 3
        r[r3] = x[r3] ** (-beta) * t2 ** beta
    return r