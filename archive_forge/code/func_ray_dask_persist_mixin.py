import dask
from .scheduler import (
from .callbacks import (
from .optimizations import dataframe_optimize
def ray_dask_persist_mixin(self, **kwargs):
    kwargs['ray_persist'] = True
    return dask_persist_mixin(self, **kwargs)