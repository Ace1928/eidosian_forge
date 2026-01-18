import numpy as np
from .. import util
def is_dask(array):
    da = dask_array_module()
    if da is None:
        return False
    return da and isinstance(array, da.Array)