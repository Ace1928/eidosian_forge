import numpy as np
import math
from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
def select_step(v1, v2, nv, hour=False, include_last=True, threshold_factor=3600.0):
    if v1 > v2:
        v1, v2 = (v2, v1)
    dv = (v2 - v1) / nv
    if hour:
        _select_step = select_step_hour
        cycle = 24.0
    else:
        _select_step = select_step_degree
        cycle = 360.0
    if dv > 1 / threshold_factor:
        step, factor = _select_step(dv)
    else:
        step, factor = select_step_sub(dv * threshold_factor)
        factor = factor * threshold_factor
    levs = np.arange(np.floor(v1 * factor / step), np.ceil(v2 * factor / step) + 0.5, dtype=int) * step
    n = len(levs)
    if factor == 1.0 and levs[-1] >= levs[0] + cycle:
        nv = int(cycle / step)
        if include_last:
            levs = levs[0] + np.arange(0, nv + 1, 1) * step
        else:
            levs = levs[0] + np.arange(0, nv, 1) * step
        n = len(levs)
    return (np.array(levs), n, factor)