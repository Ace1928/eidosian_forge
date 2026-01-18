import numpy as np
from scipy import sparse
from ..utils.extmath import squared_norm
def l2_penalty(self, weights, l2_reg_strength):
    """Compute L2 penalty term l2_reg_strength/2 *||w||_2^2."""
    norm2_w = weights @ weights if weights.ndim == 1 else squared_norm(weights)
    return 0.5 * l2_reg_strength * norm2_w