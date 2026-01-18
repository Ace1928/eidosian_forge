import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def calc_wilcoxon_test(baseline, test):
    return compute_wx_test(baseline, test)['pvalue']