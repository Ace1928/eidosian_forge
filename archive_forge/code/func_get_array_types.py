import numpy as np
from .. import util
def get_array_types():
    array_types = (np.ndarray,)
    da = dask_array_module()
    if da is not None:
        array_types += (da.Array,)
    return array_types