import numpy as np
import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
def forward_time(xi, yi):
    if not dmap.grid.within_grid(xi, yi):
        raise OutOfBounds
    ds_dt = interpgrid(speed, xi, yi)
    if ds_dt == 0:
        raise TerminateTrajectory()
    dt_ds = 1.0 / ds_dt
    ui = interpgrid(u, xi, yi)
    vi = interpgrid(v, xi, yi)
    return (ui * dt_ds, vi * dt_ds)