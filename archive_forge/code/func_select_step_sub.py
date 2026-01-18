import numpy as np
import math
from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
def select_step_sub(dv):
    tmp = 10.0 ** (int(math.log10(dv)) - 1.0)
    factor = 1.0 / tmp
    if 1.5 * tmp >= dv:
        step = 1
    elif 3.0 * tmp >= dv:
        step = 2
    elif 7.0 * tmp >= dv:
        step = 5
    else:
        step = 1
        factor = 0.1 * factor
    return (step, factor)