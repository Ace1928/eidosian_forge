import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def calc_bootstrap_ci_for_mean(samples, level=0.05, tries=999):
    """
    Count confidence intervals for difference each two samples.

    Args:
        :param samples: samples
        :param level: (float) Level for the confidence interval.
        :param tries: bootstrap samples to use
        :return: (left, right) border of confidence interval

    """
    if not (samples == 0).all():
        samples = np.array(samples)
        means = []
        for _ in range(0, tries):
            resample = np.random.choice(samples, len(samples))
            means.append(np.mean(resample))
        means = sorted(means)
        left = means[int(tries * (level / 2))]
        right = means[int(tries * (1.0 - level / 2))]
        return (left, right)
    else:
        return (0, 0)