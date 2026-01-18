import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import numpy as np
from .AxisItem import AxisItem
def skipFactor(self, minSpc):
    if self.autoSkip is None or minSpc < self.spacing:
        return 1
    factors = np.array(self.autoSkip, dtype=np.float64)
    while True:
        for f in factors:
            spc = self.spacing * f
            if spc > minSpc:
                return int(f)
        factors *= 10