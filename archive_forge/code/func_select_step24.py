import numpy as np
import math
from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
def select_step24(v1, v2, nv, include_last=True, threshold_factor=3600):
    v1, v2 = (v1 / 15, v2 / 15)
    levs, n, factor = select_step(v1, v2, nv, hour=True, include_last=include_last, threshold_factor=threshold_factor)
    return (levs * 15, n, factor)