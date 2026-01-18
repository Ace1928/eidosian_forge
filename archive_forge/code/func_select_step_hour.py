import numpy as np
import math
from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
def select_step_hour(dv):
    hour_limits_ = [1.5, 2.5, 3.5, 5, 7, 10, 15, 21, 36]
    hour_steps_ = [1, 2, 3, 4, 6, 8, 12, 18, 24]
    hour_factors = [1.0] * len(hour_steps_)
    minsec_limits_ = [1.5, 2.5, 3.5, 4.5, 5.5, 8, 11, 14, 18, 25, 45]
    minsec_steps_ = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30]
    minute_limits_ = np.array(minsec_limits_) / 60
    minute_factors = [60.0] * len(minute_limits_)
    second_limits_ = np.array(minsec_limits_) / 3600
    second_factors = [3600.0] * len(second_limits_)
    hour_limits = [*second_limits_, *minute_limits_, *hour_limits_]
    hour_steps = [*minsec_steps_, *minsec_steps_, *hour_steps_]
    hour_factors = [*second_factors, *minute_factors, *hour_factors]
    n = np.searchsorted(hour_limits, dv)
    step = hour_steps[n]
    factor = hour_factors[n]
    return (step, factor)