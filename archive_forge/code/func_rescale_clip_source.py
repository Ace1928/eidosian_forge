import numba
import numpy as np
def rescale_clip_source(xx, scale, offset, vmin, vmax, yy):
    for i in range(xx.size):
        val = (xx[i] - offset) * scale
        yy[i] = min(max(val, vmin), vmax)