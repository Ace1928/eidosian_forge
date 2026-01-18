import numpy as np
import math
from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
def select_step360(v1, v2, nv, include_last=True, threshold_factor=3600):
    return select_step(v1, v2, nv, hour=False, include_last=include_last, threshold_factor=threshold_factor)