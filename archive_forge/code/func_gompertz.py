import numpy as np
def gompertz(x, A, u, d, v, y0):
    """Gompertz growth model.

    Proposed in Zwietering et al., 1990 (PMID: 16348228)
    """
    y = A * np.exp(-np.exp(u * np.e / A * (d - x) + 1)) + y0
    return y