import numpy as np
import xarray as xr
def np_to_png(a):
    if 2 <= len(a.shape) <= 3:
        return fromarray(np.array(np.clip(a, 0, 1) * 255, dtype='uint8'))._repr_png_()
    return fromarray(np.zeros([1, 1], dtype='uint8'))._repr_png_()