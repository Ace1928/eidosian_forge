import numbers
import numpy as np
from scipy.special import xlogy
from ..utils import check_scalar
from ..utils.stats import _weighted_percentile
from ._loss import (
from .link import (
def in_y_pred_range(self, y):
    """Return True if y is in the valid range of y_pred.

        Parameters
        ----------
        y : ndarray
        """
    return self.interval_y_pred.includes(y)