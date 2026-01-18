import numpy as np
import xarray as xr
def np_to_text(obj, p, cycle):
    if len(obj.shape) < 2:
        print(repr(obj))
    if 2 <= len(obj.shape) <= 3:
        pass
    else:
        print(f'<array of shape {obj.shape}>')