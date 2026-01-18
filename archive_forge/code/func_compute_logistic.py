import numpy as np
def compute_logistic(val: float) -> float:
    v = 1.0 / (1.0 + np.exp(-np.abs(val)))
    return 1.0 - v if val < 0 else v