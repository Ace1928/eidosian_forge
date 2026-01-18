import numbers
import numpy as np
from scipy.special import xlogy
from ..utils import check_scalar
from ..utils.stats import _weighted_percentile
from ._loss import (
from .link import (
def fit_intercept_only(self, y_true, sample_weight=None):
    """Compute raw_prediction of an intercept-only model.

        This is the softmax of the weighted average of the target, i.e. over
        the samples axis=0.
        """
    out = np.zeros(self.n_classes, dtype=y_true.dtype)
    eps = np.finfo(y_true.dtype).eps
    for k in range(self.n_classes):
        out[k] = np.average(y_true == k, weights=sample_weight, axis=0)
        out[k] = np.clip(out[k], eps, 1 - eps)
    return self.link.link(out[None, :]).reshape(-1)