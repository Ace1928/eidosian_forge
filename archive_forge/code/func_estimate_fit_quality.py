import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def estimate_fit_quality(self):
    """

        :return: Simple sanity check that all models overfit and not too fast
        """
    count_overfitting, count_underfitting = self.count_under_and_over_fits()
    if count_overfitting > count_underfitting:
        return 'Overfitting'
    if count_underfitting > count_overfitting:
        return 'Underfitting'
    return 'Good'