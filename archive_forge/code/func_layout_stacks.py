import numpy as np
from ..rcparams import rcParams
from .plot_utils import get_plotting_function
def layout_stacks(stack_locs, stack_counts, binwidth, stackratio, rotated):
    """Use count and location of stacks to get coordinates of dots."""
    dotheight = stackratio * binwidth
    binradius = binwidth / 2
    x = np.repeat(stack_locs, stack_counts)
    y = np.hstack([dotheight * np.arange(count) + binradius for count in stack_counts])
    if rotated:
        x, y = (y, x)
    return (x, y)