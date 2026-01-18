import numpy as np
def make_ensemble(N=1000, rng=np.random):
    M = np.array([(0.066, -0.812, 1.996), (0.055, 0.206, 0.082), (-0.034, 0.007, 0.004)])
    alpha = rng.normal(0.0, 1.0, (N, 3))
    return c0 + np.dot(alpha, M)