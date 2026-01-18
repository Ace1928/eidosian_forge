import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def get_best_metrics(self):
    """

        :return: pandas series with best metric values
        """
    return self._fold_metric