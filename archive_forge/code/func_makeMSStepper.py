import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import numpy as np
from .AxisItem import AxisItem
def makeMSStepper(stepSize):

    def stepper(val, n, first: bool):
        if val < MIN_REGULAR_TIMESTAMP or val > MAX_REGULAR_TIMESTAMP:
            return np.inf
        if first:
            val *= 1000
            f = stepSize * 1000
            return (val // (n * f) + 1) * (n * f) / 1000.0
        else:
            return val + n * stepSize
    return stepper