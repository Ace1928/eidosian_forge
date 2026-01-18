import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def get_baseline_comparison(self, score_config=None):
    """
        Method to get human-friendly table with model comparisons.

        Returns baseline vs all other computed cases result
        :param score_config: Config to present human-friendly score, optional. Instance of ScoreConfig
        :return: pandas DataFrame. Each row is related to one ExecutionCase.
        Each row describes how better (or worse) this case is compared to baseline.
        """
    case = self._baseline_case
    return self.get_case_comparison(case, score_config)