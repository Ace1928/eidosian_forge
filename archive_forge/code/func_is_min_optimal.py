from functools import partial
import numpy as np
from . import _catboost
def is_min_optimal(self):
    """
        Returns
        ----------
        bool :  True if metric is minimizable, False otherwise
        """
    return _catboost.is_minimizable_metric(str(self))