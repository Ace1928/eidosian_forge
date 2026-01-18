import numpy as np
def erf_inv(x: float) -> float:
    sgn = -1.0 if x < 0 else 1.0
    x = (1.0 - x) * (1 + x)
    if x == 0:
        return 0
    log = np.log(x)
    v = 2.0 / (np.pi * 0.147) + 0.5 * log
    v2 = 1.0 / 0.147 * log
    v3 = -v + np.sqrt(v * v - v2)
    x = sgn * np.sqrt(v3)
    return x