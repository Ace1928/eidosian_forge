import math
import numpy as np
from scipy.signal._wavelets import _cwt, _ricker
from scipy.stats import scoreatpercentile
from ._peak_finding_utils import (
def filt_func(line):
    if len(line[0]) < min_length:
        return False
    snr = abs(cwt[line[0][0], line[1][0]] / noises[line[1][0]])
    if snr < min_snr:
        return False
    return True