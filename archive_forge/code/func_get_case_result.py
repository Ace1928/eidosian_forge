import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def get_case_result(self, case):
    """

        :param case:
        :return: CaseEvaluationResult. Scores and other information about single execution case
        """
    return self._case_results[case]