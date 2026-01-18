import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def get_fold_curve(self, fold):
    """

        :param fold:
        :return: fold learning curve (test scores on every eval_period iteration)
        """
    return self._fold_curves[fold]