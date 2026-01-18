import numpy as np
def compute_probit(val: float) -> float:
    return 1.41421356 * erf_inv(val * 2 - 1)