import numbers
import numpy as np
from scipy.special import xlogy
from ..utils import check_scalar
from ..utils.stats import _weighted_percentile
from ._loss import (
from .link import (
def constant_to_optimal_zero(self, y_true, sample_weight=None):
    term = -2 * np.sqrt(y_true * (1 - y_true))
    if sample_weight is not None:
        term *= sample_weight
    return term